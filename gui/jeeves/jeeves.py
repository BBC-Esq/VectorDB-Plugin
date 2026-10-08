import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from core.utilities import set_cuda_paths

set_cuda_paths()

import gc
import random
import re
from ctypes import byref, c_int, sizeof, windll
from ctypes.wintypes import BOOL, DWORD, HWND
from pathlib import Path

import torch
from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

import chat.base as module_chat
from core.constants import CHAT_MODELS, JEEVES_MODELS, PROJECT_ROOT
from core.utilities import ensure_theme_config, normalize_chat_text, runs_on_this_hardware, theme_manager
from gui.web_common.web_tab import WebTab

WEB_PAGE = Path(__file__).resolve().parent / "web" / "jeeves.html"
GREETING = "Hello, my name is Jeeves. Thank you for the job opportunity! Ask me how to use this program."
JEEVES_RAG_INSTRUCTION = (
    "The excerpts below are from this program's user guide. Answer the user's question using them "
    "as your source. Give a direct, specific, helpful answer; if the excerpts only partially cover "
    "the question, still give the most useful answer you can from them. Never tell the user that the "
    "excerpts do not address the question; simply answer as best you can, in character as the butler."
)
VOICES = ['bm_george', 'bm_lewis', 'bf_isabella', 'af']
SPEEDS = {'Slow': 1.0, 'Medium': 1.3, 'Fast': 1.6}
POEM_OFFERS = [
    "If I may be so bold, sir or madam -- might I interest you in a poem to pass the time?",
    "Pardon the intrusion, but might I recite a poem for your enjoyment?",
    "If you can spare a moment, perhaps a spot of poetry would be agreeable?",
]
TYPE_INTERVAL_MS = 22


def load_poems():
    poems = []
    path = PROJECT_ROOT / "Assets" / "jeeves_poems.txt"
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        for block in raw.split("@@@@@"):
            block = block.strip("\n").rstrip()
            if not block.strip():
                continue
            lines = block.split("\n")
            title = lines[0].strip().strip('"' + chr(0x201c) + chr(0x201d))
            author = ""
            for line in lines[1:]:
                if line.strip():
                    author = re.sub(r'^(?:[Bb]y[:\s]+)', '', line.strip()).strip()
                    break
            label = f"{title} - {author}" if author else title
            poems.append({"title": title, "label": label, "text": block})
    return poems


def source_rows(contexts, metadata):
    rows = []
    for i, chunk in enumerate(contexts):
        meta = metadata[i] if i < len(metadata) else {}
        path = str(meta.get('file_path') or '')
        text = (chunk or '').strip()
        lines = text.split("\n", 1)
        title = lines[0].lstrip("#").strip() if lines and lines[0].lstrip().startswith("#") else ""
        body = lines[1].strip() if title and len(lines) > 1 else text
        body = re.sub(r"[ \t]*(?<!\n)\n(?!\n)[ \t]*", " ", body)
        if len(body) > 700:
            body = body[:700] + ' ...'
        score = meta.get('similarity_score')
        rows.append({
            "title": title or meta.get('file_name') or os.path.basename(path) or 'unknown source',
            "path": path,
            "score": score if isinstance(score, (int, float)) else None,
            "text": body,
        })
    return rows


def speech_text(text):
    return normalize_chat_text(re.sub(r'#{2,}', '', text.replace('*', '')))


class GenerationWorker(QThread):
    token_signal = Signal(str)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, model_instance, augmented_query):
        super().__init__()
        self.model_instance = model_instance
        self.augmented_query = augmented_query
        self._is_running = True

    def run(self):
        try:
            for chunk in module_chat.generate_response(self.model_instance, self.augmented_query):
                if not self._is_running:
                    break
                self.token_signal.emit(chunk)
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False


class ModelLoadWorker(QThread):
    loaded_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(self, chat_model_key):
        super().__init__()
        self.chat_model_key = chat_model_key

    def run(self):
        try:
            from chat.jeeves_model import load_jeeves_model
            self.loaded_signal.emit(load_jeeves_model(self.chat_model_key))
        except Exception as e:
            self.error_signal.emit(str(e))


class SearchWorker(QThread):
    results_signal = Signal(list, list)
    error_signal = Signal(str)

    def __init__(self, vector_db, query):
        super().__init__()
        self.vector_db = vector_db
        self.query = query

    def run(self):
        try:
            contexts, metadata = self.vector_db.search(self.query, k=5, score_threshold=0.5, search_term="", document_types="")
            self.results_signal.emit(contexts, metadata)
        except Exception as e:
            self.error_signal.emit(str(e))


class TTSWorker(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, tts, text, voice, speed):
        super().__init__()
        self.tts = tts
        self.text = text
        self.voice = voice
        self.speed = speed
        self._should_stop = False

    def stop(self):
        self._should_stop = True
        if hasattr(self.tts, 'stop'):
            self.tts.stop()

    def run(self):
        try:
            if not self._should_stop:
                self.tts.speak(speech_text(self.text), voice=self.voice, speed=self.speed)
            self.finished.emit()
        except Exception as e:
            if not self._should_stop:
                self.error.emit(str(e))
            else:
                self.finished.emit()


class JeevesPage(WebTab):
    def __init__(self, parent=None):
        super().__init__(WEB_PAGE, parent)
        self.models = []
        for name in JEEVES_MODELS:
            info = CHAT_MODELS.get(name)
            if info and runs_on_this_hardware(info):
                self.models.append({"name": name, "memory": round(info["vram"] / 1024, 1) if torch.cuda.is_available() else None,
                                    "downloaded": self.model_downloaded(info)})
        self.model = None
        self.loaded = None
        self.loading = None
        self.model_instance = None
        self.messages = []
        self._next_id = 1
        self.phase = "idle"
        self.prompt = None
        self.notice = None
        self.poems = load_poems()
        self.voice = 'bm_george'
        self.speed = 'Medium'
        self.speaking = None
        self.worker = None
        self._load_worker = None
        self._search_worker = None
        self._answer_id = None
        self._question = None
        self.tts_thread = None
        self.tts_worker = None
        self._typing = []
        self._typer = QTimer(self)
        self._typer.setInterval(TYPE_INTERVAL_MS)
        self._typer.timeout.connect(self._type_step)
        self.add_message("jeeves", GREETING, kind="greeting")
        self.vector_db = None
        self.db_error = None
        try:
            self.vector_db = self.open_database()
        except Exception as e:
            self.db_error = str(e)
        self.tts = None
        try:
            self.tts = self.make_tts()
        except Exception:
            self.tts = None

    def model_downloaded(self, info):
        folder = PROJECT_ROOT / "Models" / "chat" / info["cache_dir"]
        try:
            return folder.is_dir() and any(folder.iterdir())
        except OSError:
            return False

    def open_database(self):
        from db.database_interactions import get_query_db

        return get_query_db("user_manual")

    def make_tts(self):
        from modules.kokoro import KokoroTTS

        return KokoroTTS(repo_path=str(PROJECT_ROOT / "Models" / "tts" / "ctranslate2-4you--Kokoro-82M-light"))

    def build_state(self):
        image = PROJECT_ROOT / "Assets" / "ask_jeeves_transparent.jpg"
        return {
            "image": QUrl.fromLocalFile(str(image)).toString() if image.exists() else None,
            "models": self.models,
            "model": self.model,
            "loaded": self.loaded,
            "loading": self.loading,
            "messages": self.messages,
            "phase": self.phase,
            "prompt": self.prompt,
            "poems": [p["label"] for p in self.poems] if self.prompt and self.prompt["type"] == "choose" else [],
            "notice": self.notice,
            "db_error": self.db_error,
            "tts": self.tts is not None,
            "voices": VOICES,
            "voice": self.voice,
            "speeds": list(SPEEDS),
            "speed": self.speed,
            "speaking": self.speaking,
        }

    def add_message(self, role, text, kind="answer", streaming=False):
        message = {"id": self._next_id, "role": role, "text": text, "kind": kind, "streaming": streaming, "sources": []}
        self._next_id += 1
        self.messages.append(message)
        self.schedule_push()
        return message

    def _message(self, message_id):
        return next((m for m in self.messages if m["id"] == message_id), None)

    def busy(self):
        return self.phase != "idle" or self.prompt is not None or self.loading is not None

    def js_select_model(self, name):
        if self.phase != "idle" or self.loading:
            return {"error": "Wait for Jeeves to finish first."}
        if not any(m["name"] == name for m in self.models):
            return {"error": "That model is not available on this computer."}
        self.unload()
        self.model = name
        self.loading = name
        self.notice = None
        self._load_worker = ModelLoadWorker(name)
        self._load_worker.loaded_signal.connect(self._on_model_loaded)
        self._load_worker.error_signal.connect(self._on_load_error)
        self._load_worker.start()
        self.schedule_push()
        return {"ok": True}

    def _finish_load_worker(self):
        if self._load_worker:
            self._load_worker.wait()
            self._load_worker = None

    def _on_model_loaded(self, model_instance):
        self._finish_load_worker()
        self.model_instance = model_instance
        self.loaded = self.loading
        self.loading = None
        for m in self.models:
            if m["name"] == self.loaded:
                m["downloaded"] = True
        self.schedule_push()

    def _on_load_error(self, message):
        self._finish_load_worker()
        self.notice = {"kind": "error", "message": f"Could not load the model: {message}"}
        self.loading = None
        self.model = None
        self.schedule_push()

    def unload(self):
        if self.model_instance:
            try:
                self.model_instance.cleanup()
            except Exception:
                pass
            self.model_instance = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.loaded = None

    def js_eject(self):
        if self.phase != "idle":
            return {"error": "Wait for Jeeves to finish first."}
        self.unload()
        self.model = None
        self.schedule_push()
        return {"ok": True}

    def js_ask(self, text):
        text = (text or "").strip()
        if not text:
            return {"error": "Type a question first."}
        if self.loading:
            return {"error": "Wait for the model to finish loading."}
        if not self.model_instance:
            return {"error": "Please select a language model before sending a message."}
        if self.busy():
            return {"error": "Jeeves is busy."}
        if self.vector_db is None:
            return {"error": f"The user manual database could not be loaded, so Jeeves cannot answer questions. {self.db_error or ''}".strip()}
        self.notice = None
        self.add_message("user", text, kind="question")
        self._question = text
        self.phase = "searching"
        self._search_worker = SearchWorker(self.vector_db, text)
        self._search_worker.results_signal.connect(self._on_search_results)
        self._search_worker.error_signal.connect(self._on_search_error)
        self._search_worker.start()
        self.schedule_push()
        return {"ok": True}

    def _finish_search_worker(self):
        if self._search_worker:
            self._search_worker.wait()
            self._search_worker = None

    def _on_search_error(self, message):
        self._finish_search_worker()
        self.phase = "idle"
        self.notice = {"kind": "error", "message": f"An error occurred while querying the database: {message}"}
        self.schedule_push()

    def _on_search_results(self, contexts, metadata):
        self._finish_search_worker()
        if not contexts:
            self.phase = "idle"
            self.notice = {"kind": "warn", "message": "No relevant chunks were found in the user manual database for this question. Try rephrasing your question."}
            self.schedule_push()
            return
        if not self.model_instance:
            self.phase = "idle"
            self.schedule_push()
            return
        contexts_text = "\n\n".join(contexts)
        augmented_query = (
            f"{JEEVES_RAG_INSTRUCTION}\n\n"
            f"USER GUIDE EXCERPTS:\n{contexts_text}\n\n"
            f"QUESTION: {self._question}"
        )
        answer = self.add_message("jeeves", "", kind="answer", streaming=True)
        answer["sources"] = source_rows(contexts, metadata)
        self._answer_id = answer["id"]
        self.phase = "answering"
        self.worker = GenerationWorker(self.model_instance, augmented_query)
        self.worker.token_signal.connect(self._on_token)
        self.worker.error_signal.connect(self._on_generation_error)
        self.worker.finished_signal.connect(self._on_generation_finished)
        self.worker.start()
        self.schedule_push()

    def _on_token(self, token):
        message = self._message(self._answer_id)
        if message is not None:
            message["text"] += token
            self.schedule_push()

    def _finish_generation(self):
        if self.worker:
            if self.worker.isRunning():
                self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
        message = self._message(self._answer_id)
        if message is not None:
            message["streaming"] = False
        self._answer_id = None

    def _on_generation_error(self, error):
        message = self._message(self._answer_id)
        self._finish_generation()
        if message is not None and not message["text"].strip():
            self.messages.remove(message)
        self.phase = "idle"
        self.notice = {"kind": "error", "message": f"An error occurred: {error}"}
        self.schedule_push()

    def _on_generation_finished(self):
        self._finish_generation()
        self.phase = "idle"
        if self.poems and self.poem_roll():
            self.offer_poem()
        self.schedule_push()

    def poem_roll(self):
        return random.random() < (1.0 / 3.0)

    def type_message(self, text, kind, then=None):
        message = self.add_message("jeeves", "", kind=kind, streaming=True)
        self._typing.append((message["id"], re.findall(r'\S+|\s+', text), then))
        self.phase = "typing"
        if not self._typer.isActive():
            self._typer.start()
        return message

    def _type_step(self):
        if not self._typing:
            self._typer.stop()
            return
        message_id, tokens, then = self._typing[0]
        message = self._message(message_id)
        while tokens and not tokens[0].strip():
            if message is not None:
                message["text"] += tokens.pop(0)
            else:
                tokens.pop(0)
        if tokens:
            token = tokens.pop(0)
            if message is not None:
                message["text"] += token
        if not tokens:
            self._typing.pop(0)
            if message is not None:
                message["streaming"] = False
            if not self._typing:
                self._typer.stop()
                self.phase = "idle"
            if then:
                then()
        self.push_state()

    def offer_poem(self):
        def ask():
            self.prompt = {"type": "offer"}
            self.schedule_push()
        self.type_message(random.choice(POEM_OFFERS), "offer", ask)

    def js_poem_answer(self, yes):
        if not self.prompt or self.prompt["type"] != "offer":
            return {"ok": True}
        self.prompt = None
        if yes:
            def choose():
                self.prompt = {"type": "choose"}
                self.schedule_push()
            self.type_message("Splendid! Which poem shall I recite for you? Kindly make your selection below.", "offer", choose)
        else:
            self.type_message("Very good, sir. Perhaps another time.", "info")
        return {"ok": True}

    def js_poem_recite(self, index):
        if not self.prompt or self.prompt["type"] != "choose":
            return {"ok": True}
        self.prompt = None
        if not 0 <= int(index) < len(self.poems):
            self.schedule_push()
            return {"ok": True}
        self.type_message(self.poems[int(index)]["text"], "poem")
        return {"ok": True}

    def js_poem_cancel(self):
        if not self.prompt:
            return {"ok": True}
        self.prompt = None
        self.type_message("Very good, sir.", "info")
        return {"ok": True}

    def js_set_voice(self, value):
        if value in VOICES:
            self.voice = value
            self.schedule_push()
        return {"ok": True}

    def js_set_speed(self, value):
        if value in SPEEDS:
            self.speed = value
            self.schedule_push()
        return {"ok": True}

    def js_speak(self, message_id):
        if self.tts is None:
            return {"error": "Text-to-speech is not available. Please check if KokoroTTS is properly installed."}
        if self.speaking is not None:
            return {"error": "Jeeves is already speaking."}
        message = self._message(message_id)
        if message is None or not message["text"].strip() or message["streaming"]:
            return {"error": "The response is empty. Please ask a question first."}
        self.speaking = message_id
        self.tts_thread = QThread()
        self.tts_worker = TTSWorker(self.tts, message["text"], self.voice, SPEEDS[self.speed])
        self.tts_worker.moveToThread(self.tts_thread)
        self.tts_thread.started.connect(self.tts_worker.run)
        self.tts_worker.finished.connect(self._on_speech_finished)
        self.tts_worker.error.connect(self._on_speech_error)
        self.tts_thread.start()
        self.schedule_push()
        return {"ok": True}

    def js_stop_speaking(self):
        if self.tts_worker:
            self.tts_worker.stop()
        return {"ok": True}

    def _on_speech_finished(self):
        self.speaking = None
        if self.tts_thread:
            self.tts_thread.quit()
            self.tts_thread.wait()
            self.tts_thread = None
        if self.tts_worker:
            self.tts_worker.deleteLater()
            self.tts_worker = None
        self.schedule_push()

    def _on_speech_error(self, message):
        self._on_speech_finished()
        self.notice = {"kind": "error", "message": f"An error occurred while trying to speak: {message}"}
        self.schedule_push()

    def js_open_source(self, path):
        if path.startswith(("http://", "https://")):
            QDesktopServices.openUrl(QUrl(path))
            return {"ok": True}
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            return {"ok": True}
        self.notice = {"kind": "warn", "message": f"Could not open the source file: {path}"}
        self.schedule_push()
        return {"missing": True}

    def js_clear(self):
        if self.phase != "idle":
            return {"error": "Wait for Jeeves to finish first."}
        self.prompt = None
        self.messages = []
        self.add_message("jeeves", GREETING, kind="greeting")
        self.notice = None
        return {"ok": True}

    def js_dismiss_notice(self):
        self.notice = None
        self.schedule_push()
        return {"ok": True}

    def cleanup(self):
        self._typer.stop()
        for worker in (self.worker, self._load_worker, self._search_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, 'stop'):
                    worker.stop()
                worker.wait(5000)
        if self.tts_worker is not None:
            self.tts_worker.stop()
        if self.tts_thread is not None and self.tts_thread.isRunning():
            self.tts_thread.quit()
            self.tts_thread.wait(5000)
        if self.vector_db is not None:
            self.vector_db.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ChatWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ask Jeeves (Welcome back Jeeves!)")
        self.setGeometry(100, 100, 850, 950)
        self.page = JeevesPage(self)
        self.setCentralWidget(self.page)

    def showEvent(self, event):
        super().showEvent(event)
        self.apply_dark_mode_settings()

    def apply_dark_mode_settings(self):
        set_window_attribute = windll.dwmapi.DwmSetWindowAttribute
        hwnd = HWND(int(self.winId()))
        true_bool = BOOL(True)
        set_window_attribute(hwnd, DWORD(20), byref(true_bool), sizeof(true_bool))
        black_color = c_int(0xFF000000)
        set_window_attribute(hwnd, DWORD(34), byref(black_color), sizeof(black_color))

    def closeEvent(self, event):
        self.page.cleanup()
        event.accept()


def launch_jeeves_process():
    from core.utilities import set_cuda_paths
    set_cuda_paths()
    from PySide6.QtCore import Qt

    flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "--process-per-site" not in flags:
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{flags} --process-per-site".strip()
    if hasattr(QApplication, 'setHighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication([])
    theme_manager.apply(ensure_theme_config())

    try:
        window = ChatWindow()
    except Exception as e:
        print(f"Jeeves failed to start: {e}")
        QMessageBox.critical(None, "Ask Jeeves", f"Jeeves could not start:\n{e}")
        return
    window.show()

    ret = app.exec()
    sys.exit(ret)
