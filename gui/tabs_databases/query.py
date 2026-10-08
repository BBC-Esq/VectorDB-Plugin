import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMessageBox, QTabWidget

from chat.kobold import KoboldThread
from chat.lm_studio import LMStudioChatThread
from chat.local_model import LocalModelChat
from chat.minimax import MiniMaxThread
from chat.openai import ChatGPTThread
from core.constants import PROJECT_ROOT
from core.utilities import cuda_usable, normalize_chat_text, open_file, runs_on_this_hardware
from db.process_manager import get_process_manager
from gui.web_common.web_tab import WebTab
from gui.tabs_databases import query_data
from gui.tabs_databases.query_search import chunks_query
from modules.tts_process import run_tts_in_process

WEB_PAGE = Path(__file__).resolve().parent / "web" / "query_tab.html"
HISTORY_LIMIT = 50
CHUNKS_TIMEOUT = 120


def chat_history_path():
    return PROJECT_ROOT / "chat_history.txt"


def kill_process_tree(pid):
    import psutil

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        procs = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        procs = []
    procs.append(parent)
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def reveal_file(path):
    if path and os.path.exists(path):
        if sys.platform == "win32":
            subprocess.Popen(f'explorer /select,"{os.path.normpath(path)}"')
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))
        return {"ok": True}
    parent = Path(path).parent if path else None
    if parent and parent.is_dir():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(parent)))
        return {"missing": True, "opened_folder": True}
    return {"missing": True}


class TurnRelay(QObject):
    def __init__(self, tab, turn_id):
        super().__init__(tab)
        self.tab = tab
        self.turn_id = turn_id

    @Slot(str)
    def response(self, text):
        self.tab.on_response(self.turn_id, text)

    @Slot(str)
    def citations(self, html):
        self.tab.on_citations(self.turn_id, html)

    @Slot(str)
    def error(self, message):
        self.tab.on_error(self.turn_id, message)

    @Slot()
    def finished(self):
        self.tab.on_finished(self.turn_id)

    @Slot(object)
    def chunks(self, result):
        self.tab.on_chunks(self.turn_id, result)

    @Slot(str)
    def failed(self, message):
        self.tab.on_error(self.turn_id, message)
        self.tab.on_finished(self.turn_id)


class LocalSubmitWorker(QThread):
    failed = Signal(str)

    def __init__(self, chat, model, question, database, parent=None):
        super().__init__(parent)
        self.chat = chat
        self.model = model
        self.question = question
        self.database = database

    def run(self):
        try:
            if self.chat.current_model != self.model or not self.chat.is_model_loaded():
                if self.chat.model_process is not None or self.chat.current_model is not None:
                    self.chat.terminate_current_process()
                self.chat.start_model_process(self.model)
            self.chat.start_chat(self.question, self.model, self.database)
        except Exception as e:
            logging.exception("Starting the local model failed")
            self.failed.emit(str(e))


class EjectWorker(QThread):
    def __init__(self, chat, parent=None):
        super().__init__(parent)
        self.chat = chat

    def run(self):
        try:
            self.chat.eject_model()
        except Exception:
            logging.exception("Ejecting the local model failed")


class ChunksWorker(QThread):
    result = Signal(object)

    def __init__(self, question, database, parent=None):
        super().__init__(parent)
        self.question = question
        self.database = database
        self.process = None
        self._lock = threading.Lock()
        self._stop = False

    def run(self):
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        try:
            with self._lock:
                if self._stop:
                    return
                self.process = ctx.Process(target=chunks_query, args=(self.database, self.question, result_queue))
                get_process_manager().register(self.process)
                self.process.start()
                process = self.process
            self.result.emit(self.wait_for_result(result_queue, process))
        except Exception as e:
            logging.exception("Chunks only query failed")
            self.result.emit({"error": f"The database could not be searched: {e}"})
        finally:
            self.stop_process()

    def wait_for_result(self, result_queue, process):
        deadline = time.monotonic() + CHUNKS_TIMEOUT
        while not self._stop:
            try:
                return result_queue.get(timeout=0.5)
            except queue.Empty:
                if not process.is_alive():
                    try:
                        return result_queue.get(timeout=1)
                    except queue.Empty:
                        return {"error": "The database search stopped unexpectedly. Check the command prompt window for details."}
                if time.monotonic() >= deadline:
                    return {"error": f"The database search took longer than {CHUNKS_TIMEOUT} seconds and was stopped."}
        return {"error": "The search was stopped."}

    def stop_process(self):
        with self._lock:
            process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.is_alive():
                process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        except Exception:
            logging.exception("Stopping the search process failed")
        finally:
            get_process_manager().unregister(process)

    def stop(self):
        self._stop = True
        self.stop_process()


class DatabaseQueryTab(WebTab):
    def __init__(self, parent=None):
        super().__init__(WEB_PAGE, parent)
        self.local_model_chat = LocalModelChat()
        self.turns = []
        self._next_turn = 1
        self.active = None
        self._local_turn = None
        self._threads = {}
        self._relays = {}
        self._workers = set()
        self.chunks_only = False
        self.backend = "Local Model"
        self._config = {}
        self.databases = []
        self.database = None
        self.models = []
        self.local_model = None
        self.ejecting = False
        self.speaking = None
        self._tts_process = None
        self._tts_timer = None
        self._tts_stopped = False
        self.voice = "idle"
        self.voice_started = None
        self.transcript = None
        self._voice_recorder = None
        self.notice = None
        self.focus = 0
        self._stamp = None
        signals = self.local_model_chat.signals
        signals.response_signal.connect(self._on_local_response)
        signals.citations_signal.connect(self._on_local_citations)
        signals.error_signal.connect(self._on_local_error)
        signals.finished_signal.connect(self._on_local_finished)
        signals.token_count_signal.connect(self._on_local_tokens)
        signals.model_loaded_signal.connect(self.schedule_push)
        signals.model_unloaded_signal.connect(self.schedule_push)
        self._poll = QTimer(self)
        self._poll.setInterval(1000)
        self._poll.timeout.connect(self._tick)
        self.refresh()

    def showEvent(self, event):
        self.refresh()
        self._poll.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._poll.stop()
        super().hideEvent(event)

    def _stamps(self):
        stamps = []
        for path in (query_data.config_path(), PROJECT_ROOT / "Vector_DB", PROJECT_ROOT / "Models" / "chat"):
            try:
                st = path.stat()
                stamps.append((st.st_mtime_ns, st.st_size))
            except OSError:
                stamps.append(None)
        return tuple(stamps)

    def _tick(self):
        if self._stamps() != self._stamp:
            self.refresh()

    def refresh(self):
        self._stamp = self._stamps()
        self._config = query_data.load_config()
        self.databases = query_data.queryable_databases(self._config)
        names = [d["name"] for d in self.databases]
        if self.database not in names:
            self.database = query_data.remembered_database(self._config, names)
        self.models = query_data.local_models(self._config)
        model_names = [m["name"] for m in self.models]
        if self.local_model not in model_names:
            self.local_model = model_names[0] if model_names else None
        self.schedule_push()

    def _turn(self, turn_id):
        return next((t for t in self.turns if t["id"] == turn_id), None)

    def _readiness(self):
        if self.chunks_only:
            return None
        return query_data.readiness(self._config, self.backend, self.local_model, self.models)

    def loaded_model(self):
        return self.local_model_chat.current_model if self.local_model_chat.is_model_loaded() else None

    def build_state(self):
        return {
            "databases": self.databases,
            "database": self.database,
            "backends": query_data.BACKENDS,
            "backend": self.backend,
            "models": self.models,
            "local_model": self.local_model,
            "loaded_model": self.loaded_model(),
            "ejecting": self.ejecting,
            "cpu_only": not cuda_usable(),
            "settings": query_data.query_settings(self._config),
            "readiness": self._readiness(),
            "chunks_only": self.chunks_only,
            "busy": self.active is not None,
            "turns": self.turns,
            "speaking": self.speaking,
            "tts": query_data.tts_label(self._config),
            "voice": self.voice,
            "voice_started": self.voice_started,
            "transcript": self.transcript,
            "notice": self.notice,
            "focus": self.focus,
        }

    def select_database(self, name):
        if any(d["name"] == name for d in self.databases):
            self.database = name
            query_data.remember_database(name)
            self.focus += 1
            self.schedule_push()
            return True
        return False

    def js_select_database(self, name):
        if not any(d["name"] == name for d in self.databases):
            return {"error": "That database is no longer available."}
        self.database = name
        query_data.remember_database(name)
        self.schedule_push()
        return {"ok": True}

    def js_select_backend(self, name):
        if name not in query_data.BACKENDS:
            return {"error": "Unknown backend."}
        self.backend = name
        self.schedule_push()
        return {"ok": True}

    def js_select_model(self, name):
        if not any(m["name"] == name for m in self.models):
            return {"error": "That model is not available on this computer."}
        self.local_model = name
        self.schedule_push()
        return {"ok": True}

    def js_set_chunks_only(self, value):
        self.chunks_only = bool(value)
        self.schedule_push()
        return {"ok": True}

    def js_ask(self, question):
        question = (question or "").strip()
        if not question:
            return {"error": "Type a question first."}
        if self.active is not None:
            return {"error": "Wait for the current answer to finish."}
        if self.ejecting:
            return {"error": "Wait for the local model to finish unloading."}
        self.refresh()
        if not any(d["name"] == self.database for d in self.databases):
            return {"error": "Choose a database to query first."}
        problem = self._readiness()
        if problem:
            return {"error": problem["message"]}
        local = not self.chunks_only and self.backend == "Local Model"
        fresh = local and (self.local_model_chat.current_model != self.local_model or not self.local_model_chat.is_model_loaded())
        turn = {
            "id": self._next_turn,
            "kind": "chunks" if self.chunks_only else "answer",
            "question": question,
            "database": self.database,
            "backend": "Chunks only" if self.chunks_only else self.backend,
            "model": self.local_model if local else None,
            "started": time.time(),
            "ended": None,
            "phase": "loading" if fresh else "searching",
            "answer": "",
            "citations": [],
            "tokens": None,
            "error": None,
            "chunks": None,
            "similarity": query_data.query_settings(self._config)["similarity"],
        }
        self._next_turn += 1
        self.turns.append(turn)
        del self.turns[:-HISTORY_LIMIT]
        self.active = turn["id"]
        try:
            if self.chunks_only:
                self._start_chunks(turn)
            elif local:
                self._start_local(turn)
            else:
                self._start_thread(turn)
        except Exception as e:
            logging.exception("Submitting the question failed")
            self.on_error(turn["id"], str(e))
            self.on_finished(turn["id"])
        self.schedule_push()
        return {"ok": True, "turn": turn["id"]}

    def _relay(self, turn_id):
        relay = TurnRelay(self, turn_id)
        self._relays[turn_id] = relay
        return relay

    def _keep(self, worker):
        self._workers.add(worker)
        worker.finished.connect(lambda w=worker: self._workers.discard(w))

    def _start_thread(self, turn):
        relay = self._relay(turn["id"])
        question, database, backend = turn["question"], turn["database"], turn["backend"]
        if backend == "LM Studio":
            thread = LMStudioChatThread(question, database)
            signals = thread.lm_studio_chat.signals
        elif backend == "ChatGPT":
            thread = signals = ChatGPTThread(question, database)
        elif backend == "Kobold":
            thread = signals = KoboldThread(question, database)
        elif backend in query_data.MINIMAX_BACKENDS:
            thread = signals = MiniMaxThread(question, database, model_name=backend)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        signals.response_signal.connect(relay.response)
        signals.error_signal.connect(relay.error)
        signals.citations_signal.connect(relay.citations)
        signals.finished_signal.connect(relay.finished)
        self._threads[turn["id"]] = thread
        self._keep(thread)
        thread.start()

    def _start_local(self, turn):
        self._local_turn = turn["id"]
        relay = self._relay(turn["id"])
        worker = LocalSubmitWorker(self.local_model_chat, self.local_model, turn["question"], turn["database"])
        worker.failed.connect(relay.failed)
        self._keep(worker)
        worker.start()

    def _start_chunks(self, turn):
        relay = self._relay(turn["id"])
        worker = ChunksWorker(turn["question"], turn["database"])
        worker.result.connect(relay.chunks)
        self._threads[turn["id"]] = worker
        self._keep(worker)
        worker.start()

    def on_response(self, turn_id, text):
        turn = self._turn(turn_id)
        if turn is None or turn["ended"]:
            return
        turn["answer"] += text
        if turn["phase"] in ("loading", "searching"):
            turn["phase"] = "answering"
        self.schedule_push()

    def on_citations(self, turn_id, html):
        turn = self._turn(turn_id)
        if turn is None:
            return
        turn["citations"] = query_data.parse_citations(html)
        self.schedule_push()

    def on_error(self, turn_id, message):
        turn = self._turn(turn_id)
        if turn is None or turn["ended"]:
            return
        turn["error"] = str(message or "Something went wrong.").strip()
        turn["phase"] = "error"
        self.schedule_push()

    def on_finished(self, turn_id):
        turn = self._turn(turn_id)
        if turn is not None and not turn["ended"]:
            turn["ended"] = time.time()
            if turn["phase"] != "error":
                turn["phase"] = "done"
            if turn["kind"] == "answer" and turn["answer"].strip():
                self._write_chat_history(turn["answer"])
        if self.active == turn_id:
            self.active = None
        if self._local_turn == turn_id:
            self._local_turn = None
        self._threads.pop(turn_id, None)
        relay = self._relays.pop(turn_id, None)
        if relay is not None:
            relay.deleteLater()
        self.schedule_push()

    def on_chunks(self, turn_id, result):
        turn = self._turn(turn_id)
        if turn is not None and not turn["ended"]:
            result = result or {}
            if result.get("error"):
                turn["error"] = result["error"]
                turn["phase"] = "error"
            else:
                turn["chunks"] = result.get("chunks") or []
        self.on_finished(turn_id)

    def _on_local_response(self, text):
        if self._local_turn is not None:
            self.on_response(self._local_turn, text)

    def _on_local_citations(self, html):
        if self._local_turn is not None:
            self.on_citations(self._local_turn, html)

    def _on_local_error(self, message):
        if self._local_turn is not None:
            self.on_error(self._local_turn, message)
        else:
            self.notice = {"kind": "error", "message": str(message)}
            self.schedule_push()

    def _on_local_finished(self):
        if self._local_turn is not None:
            self.on_finished(self._local_turn)
        else:
            self.schedule_push()

    def _on_local_tokens(self, html):
        turn = self._turn(self._local_turn) if self._local_turn is not None else None
        if turn is not None:
            turn["tokens"] = query_data.parse_token_counts(html)
            self.schedule_push()

    def _write_chat_history(self, text):
        try:
            with open(chat_history_path(), "w", encoding="utf-8") as f:
                f.write(normalize_chat_text(text.lstrip("\n")))
            return True
        except OSError:
            logging.exception("Could not write chat_history.txt")
            return False

    def js_eject(self):
        if self.active is not None and self._local_turn is not None:
            return {"error": "Wait for the answer to finish before unloading the model."}
        if self.ejecting or not self.local_model_chat.is_model_loaded():
            return {"ok": True}
        self.ejecting = True
        worker = EjectWorker(self.local_model_chat)
        worker.finished.connect(self._on_ejected)
        self._keep(worker)
        worker.start()
        self.schedule_push()
        return {"ok": True}

    def _on_ejected(self):
        self.ejecting = False
        self.schedule_push()

    def js_copy(self, turn_id):
        turn = self._turn(turn_id)
        if turn is None:
            return {"error": "That answer is no longer listed."}
        text = query_data.turn_text(turn)
        if not text:
            return {"error": "There is nothing to copy yet."}
        QApplication.clipboard().setText(text)
        return {"ok": True}

    def js_clear(self):
        if self.active is not None:
            return {"error": "Wait for the current answer to finish."}
        self.turns = []
        self.schedule_push()
        return {"ok": True}

    def js_open_file(self, path):
        if not path or not os.path.exists(path):
            self.notice = {"kind": "warn", "message": f"{Path(path or '').name or 'The file'} is no longer at {path}. It was moved or deleted after the database was created."}
            self.schedule_push()
            return {"missing": True}
        open_file(path)
        return {"ok": True}

    def js_reveal_file(self, path):
        result = reveal_file(path)
        if result.get("missing"):
            where = " Its folder was opened instead." if result.get("opened_folder") else ""
            self.notice = {"kind": "warn", "message": f"{Path(path or '').name or 'The file'} is no longer at {path}.{where}"}
            self.schedule_push()
        return result

    def tts_preflight(self, tts_model):
        from core.constants import BACKEND_DEPENDENCIES, TTS_BACKENDS
        from core.utilities import check_backend_dependencies, download_kokoro_tts, download_with_threadpool, install_packages

        if not runs_on_this_hardware(TTS_BACKENDS.get(tts_model, {"requires_cuda": True})):
            return "The Text to Speech backend you selected requires GPU acceleration. Choose another one on the Settings tab."
        if tts_model == "kokoro":
            kokoro_dir = PROJECT_ROOT / "Models" / "tts" / "ctranslate2-4you--Kokoro-82M-light"
            if not kokoro_dir.is_dir():
                reply = QMessageBox.question(
                    self, "Kokoro TTS Model Not Found",
                    "The Kokoro TTS model is missing!\n\nWould you like to download it now?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    def on_kokoro_download_complete(success, message):
                        self.notice = ({"kind": "ok", "message": "The Kokoro TTS model was downloaded. Click Speak again to hear the answer."}
                                       if success else
                                       {"kind": "error", "message": "The Kokoro TTS model could not be downloaded. Check your internet connection and try again."})
                        self.schedule_push()

                    download_with_threadpool(download_kokoro_tts, callback=on_kokoro_download_complete)
                    return "Downloading the Kokoro TTS model…"
                return ""
        if not check_backend_dependencies(tts_model, interactive=False):
            required = BACKEND_DEPENDENCIES.get(tts_model, {})
            if required:
                packages = ", ".join(f"{pkg}=={ver}" for pkg, ver in required.items())
                reply = QMessageBox.question(
                    self, "Missing Dependencies",
                    f"{tts_model.title()} backend requires additional packages:\n\n{packages}\n\nInstall now?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply != QMessageBox.Yes:
                    return ""
                if not install_packages(list(required.items())):
                    return "The packages could not be installed. Please install them manually."
        return None

    def js_speak(self, turn_id):
        if self.speaking is not None:
            return {"error": "An answer is already being read aloud."}
        turn = self._turn(turn_id)
        if turn is None or not (turn.get("answer") or "").strip():
            return {"error": "There is no answer to read aloud."}
        self._config = query_data.load_config()
        tts_model = str((self._config.get("tts") or {}).get("model") or "").lower()
        problem = self.tts_preflight(tts_model)
        if problem is not None:
            if problem:
                self.notice = {"kind": "info" if problem.endswith("…") else "warn", "message": problem}
                self.schedule_push()
            return {"ok": False}
        if not self._write_chat_history(turn["answer"]):
            return {"error": "chat_history.txt could not be written."}
        self.start_tts(turn_id)
        return {"ok": True}

    def make_tts_process(self):
        return multiprocessing.Process(target=run_tts_in_process, args=(str(query_data.config_path()), str(chat_history_path())))

    def start_tts(self, turn_id):
        self._tts_stopped = False
        self._tts_process = self.make_tts_process()
        self._tts_process.start()
        self.speaking = turn_id
        self._tts_timer = QTimer(self)
        self._tts_timer.setInterval(400)
        self._tts_timer.timeout.connect(self._check_tts)
        self._tts_timer.start()
        self.schedule_push()

    def _check_tts(self):
        process = self._tts_process
        if process is not None and process.is_alive():
            return
        self._tts_timer.stop()
        self._tts_timer.deleteLater()
        self._tts_timer = None
        self._tts_process = None
        self.speaking = None
        if process is not None and process.exitcode and not self._tts_stopped:
            self.notice = {"kind": "error", "message": f"Text to speech stopped with an error (exit code {process.exitcode}). "
                                                        "See the command prompt window for details."}
        self.schedule_push()

    def js_stop_speaking(self):
        process = self._tts_process
        if process is not None and process.is_alive():
            self._tts_stopped = True
            kill_process_tree(process.pid)
        return {"ok": True}

    def voice_recorder(self):
        if self._voice_recorder is None:
            from modules.voice_recorder import VoiceRecorder

            self._voice_recorder = VoiceRecorder(self)
        return self._voice_recorder

    def js_record(self):
        if self.voice != "idle":
            return {"error": "The microphone is busy."}
        self.voice_recorder().start_recording()
        self.voice = "recording"
        self.voice_started = time.time()
        self.schedule_push()
        return {"ok": True}

    def js_stop_recording(self):
        if self.voice != "recording":
            return {"ok": True}
        recorder = self.voice_recorder()
        recorder.stop_recording()
        if any(t.isRunning() for t in recorder.transcription_threads):
            self.voice = "transcribing"
        else:
            self.voice = "idle"
            if self.notice is None or self.notice.get("source") != "voice":
                self.notice = {"kind": "warn", "message": "The recording was too short or silent, so nothing was transcribed.", "source": "voice"}
        self.voice_started = None
        self.schedule_push()
        return {"ok": True}

    @Slot(str)
    def update_transcription(self, text):
        text = (text or "").strip()
        failed = text.startswith("Error:") or text.startswith("[Transcription failed")
        if failed:
            self.notice = {"kind": "error", "message": text.removeprefix("Error:").strip("[] ").strip(), "source": "voice"}
        elif text:
            self.transcript = {"id": (self.transcript or {}).get("id", 0) + 1, "text": text}
        if failed or self.voice == "transcribing":
            self.voice = "idle"
            self.voice_started = None
        self.schedule_push()

    def js_dismiss_notice(self):
        self.notice = None
        self.schedule_push()
        return {"ok": True}

    def js_backend_settings(self):
        from gui.dialogs.ai_backends_dialog import AIBackendsDialog

        AIBackendsDialog(self.window(), initial_tab=query_data.SETTINGS_DIALOG_TAB.get(self.backend, 0)).exec()
        self.refresh()
        return {"ok": True}

    def open_credentials(self, kind):
        from gui.credentials import manage_credentials

        manage_credentials(self.window(), kind)
        self.refresh()
        return {"ok": True}

    def js_fix_readiness(self):
        problem = self._readiness()
        action = problem.get("action") if problem else None
        if action in ("settings", "minimax_key"):
            return self.js_backend_settings()
        if action == "hf_token":
            return self.open_credentials("hf")
        return {"ok": True}

    def js_open_tab(self, name):
        widget = self.parentWidget()
        while widget is not None and not isinstance(widget, QTabWidget):
            widget = widget.parentWidget()
        if widget is None:
            return {"error": "Tab not found."}
        for index in range(widget.count()):
            if widget.tabText(index) == name:
                widget.setCurrentIndex(index)
                return {"ok": True}
        return {"error": "Tab not found."}

    def busy_message(self):
        return None

    def cleanup(self):
        self._poll.stop()
        if self.voice == "recording" and self._voice_recorder is not None:
            self._voice_recorder.stop_recording()
        self.js_stop_speaking()
        for thread in list(self._threads.values()):
            if isinstance(thread, ChunksWorker):
                thread.stop()
            elif isinstance(thread, KoboldThread):
                thread.stop()
        if self.local_model_chat.is_model_loaded():
            self.local_model_chat.eject_model()
        for worker in list(self._workers):
            worker.wait(5000)
