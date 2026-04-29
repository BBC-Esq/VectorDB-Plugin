import logging
import queue
import threading
from pathlib import Path
import multiprocessing
import re
import html

import torch
import yaml
from PySide6.QtCore import QThread, Signal, QObject, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QPushButton, QCheckBox, QHBoxLayout, QMessageBox,
                               QApplication, QComboBox, QLabel, QTextBrowser, QProgressBar, QSizePolicy)

from abc import ABC, abstractmethod
from chat.lm_studio import LMStudioChatThread
from chat.local_model import LocalModelChat
from chat.openai import ChatGPTThread
from chat.minimax import MiniMaxThread
from chat.kobold import KoboldThread
from core.constants import CHAT_MODELS, OPENAI_MODELS, CustomButtonStyles
from modules.voice_recorder import VoiceRecorder
from core.utilities import my_cprint, normalize_chat_text
from core.constants import TOOLTIPS, PROJECT_ROOT
from db.database_interactions import process_chunks_only_query
from db.process_manager import get_process_manager

logger = logging.getLogger(__name__)

current_dir = PROJECT_ROOT
input_text_file = str(current_dir / 'chat_history.txt')

class SubmitStrategy(ABC):
    def __init__(self, tab):
        self.tab = tab

    @abstractmethod
    def submit(self, question: str, db_name: str) -> None: ...

class LocalModelStrategy(SubmitStrategy):
    def submit(self, question, db_name):
        selected_model = self.tab.model_combo_box.currentText()
        lm = self.tab.local_model_chat
        if selected_model != lm.current_model:
            if lm.is_model_loaded():
                lm.terminate_current_process()
            lm.start_model_process(selected_model)
        lm.start_chat(question, selected_model, db_name)

class LMStudioStrategy(SubmitStrategy):
    def submit(self, question, db_name):
        t = self.tab.lm_studio_chat_thread = LMStudioChatThread(question, db_name)
        s = t.lm_studio_chat.signals
        s.response_signal.connect(self.tab.update_response_lm_studio)
        s.error_signal.connect(self.tab.show_error_message)
        s.finished_signal.connect(self.tab.on_submission_finished)
        s.citations_signal.connect(self.tab.display_citations_in_widget)
        t.start()

class ChatGPTStrategy(SubmitStrategy):
    def submit(self, question, db_name):
        model_name = self.tab.model_source_combo.currentText()
        t = self.tab.chatgpt_thread = ChatGPTThread(question, db_name, model_name=model_name)
        t.response_signal.connect(self.tab.update_response_lm_studio)
        t.error_signal.connect(self.tab.show_error_message)
        t.finished_signal.connect(self.tab.on_submission_finished)
        t.citations_signal.connect(self.tab.display_citations_in_widget)
        t.start()

class MiniMaxStrategy(SubmitStrategy):
    def submit(self, question, db_name):
        model_name = self.tab.model_source_combo.currentText()
        t = self.tab.minimax_thread = MiniMaxThread(question, db_name, model_name=model_name)
        t.response_signal.connect(self.tab.update_response_lm_studio)
        t.error_signal.connect(self.tab.show_error_message)
        t.finished_signal.connect(self.tab.on_submission_finished)
        t.citations_signal.connect(self.tab.display_citations_in_widget)
        t.start()

class KoboldStrategy(SubmitStrategy):
    def submit(self, question, db_name):
        t = self.tab.kobold_thread = KoboldThread(question, db_name)
        t.response_signal.connect(self.tab.update_response_lm_studio)
        t.error_signal.connect(self.tab.show_error_message)
        t.finished_signal.connect(self.tab.on_submission_finished)
        t.citations_signal.connect(self.tab.display_citations_in_widget)
        t.start()

class ChunksOnlyStrategy(SubmitStrategy):
    def submit(self, question, db_name):
        t = self.tab.database_query_thread = ChunksOnlyThread(question, db_name)
        t.chunks_ready.connect(self.tab.display_chunks)
        t.finished.connect(self.tab.on_database_query_finished)
        t.start()

class ThinkingIndicator(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0, 0)
        self.setTextVisible(False)
        self.setFixedHeight(12)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


class ChunksOnlyThread(QThread):
    chunks_ready = Signal(str)

    def __init__(self, query, database_name):
        super().__init__()
        self.query = query
        self.database_name = database_name
        self.process = None
        self.process_lock = threading.Lock()

    def run(self):
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()

        try:
            with self.process_lock:
                self.process = ctx.Process(
                    target=process_chunks_only_query,
                    args=(self.database_name, self.query, result_queue)
                )
                get_process_manager().register(self.process)
                self.process.start()

            try:
                result = result_queue.get(timeout=120)
                self.chunks_ready.emit(result)
            except queue.Empty:
                logger.error("Query timed out after 120 seconds")
                self.chunks_ready.emit(
                    "Error: Query timed out after 120 seconds. "
                    "Please try a simpler query or check your database."
                )
            except Exception as e:
                logger.error(f"Error getting result from queue: {e}")
                self.chunks_ready.emit(f"Error: Failed to retrieve database response - {e}")

            with self.process_lock:
                if self.process and self.process.is_alive():
                    self.process.join(timeout=2)
                    if self.process.is_alive():
                        self.process.terminate()
                        self.process.join(timeout=1)
                        if self.process.is_alive():
                            try:
                                self.process.kill()
                                self.process.join(timeout=1)
                            except Exception as e:
                                logger.error(f"Failed to kill process: {e}")

                if self.process:
                    get_process_manager().unregister(self.process)
                    self.process = None

        except Exception as e:
            logger.exception(f"Error in chunks only thread: {e}")
            self.chunks_ready.emit(f"Error querying database: {e}")
            with self.process_lock:
                if self.process:
                    try:
                        if self.process.is_alive():
                            self.process.terminate()
                            self.process.join(timeout=1)
                            if self.process.is_alive():
                                self.process.kill()
                                self.process.join(timeout=1)
                        get_process_manager().unregister(self.process)
                    except Exception as cleanup_error:
                        logger.error(f"Error during cleanup: {cleanup_error}")
                    finally:
                        self.process = None

    def stop(self):
        with self.process_lock:
            if self.process:
                try:
                    if self.process.is_alive():
                        self.process.terminate()
                        self.process.join(timeout=2)
                        if self.process.is_alive():
                            self.process.kill()
                            self.process.join(timeout=1)
                    get_process_manager().unregister(self.process)
                except Exception as e:
                    logger.warning(f"Error stopping process: {e}")
                finally:
                    self.process = None


def run_tts_in_process(config_path, input_text_file):
    from modules.tts import run_tts
    run_tts(config_path, input_text_file)
    my_cprint("TTS models removed from memory.", "red")


class RefreshingComboBox(QComboBox):
    def __init__(self, parent=None):
        super(RefreshingComboBox, self).__init__(parent)

    def showPopup(self):
        new_items = self.parent().load_created_databases()
        current_items = [self.itemText(i) for i in range(self.count())]
        if new_items != current_items:
            current_text = self.currentText()
            self.clear()
            self.addItems(new_items)
            idx = self.findText(current_text)
            if idx >= 0:
                self.setCurrentIndex(idx)
        super(RefreshingComboBox, self).showPopup()


class GuiSignals(QObject):
    response_signal = Signal(str)
    citations_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()


class CustomTextBrowser(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)

    def doSetSource(self, name, type):
        if name.scheme() == 'file':
            QDesktopServices.openUrl(QUrl.fromLocalFile(name.toLocalFile()))
        elif name.scheme() in ['http', 'https']:
            QDesktopServices.openUrl(name)
        else:
            super().doSetSource(name, type)


class DatabaseQueryTab(QWidget):
    def __init__(self):
        super(DatabaseQueryTab, self).__init__()
        self.config_path = PROJECT_ROOT / 'config.yaml'
        self.lm_studio_chat_thread = None
        self.local_model_chat = LocalModelChat()
        self.chatgpt_thread = None
        self.kobold_thread = None
        self.minimax_thread = None
        self.gui_signals = GuiSignals()
        self.current_model_name = None
        self.database_query_thread = None
        self.raw_response = ""
        self.citations_block = ""
        self.in_think_block = False
        self.initWidgets()
        self.setup_signals()

    def initWidgets(self):
        layout = QVBoxLayout(self)

        self.response_widget = CustomTextBrowser()
        self.response_widget.setOpenExternalLinks(True)
        layout.addWidget(self.response_widget, 5)

        self.token_count_label = QLabel("")
        layout.addWidget(self.token_count_label)

        self.thinking_indicator = ThinkingIndicator()
        self.thinking_label = QLabel("Thinking…")
        self.thinking_label.setAlignment(Qt.AlignLeft)

        indicator_layout = QHBoxLayout()
        indicator_layout.setContentsMargins(0, 0, 0, 0)
        indicator_layout.addWidget(self.thinking_label)
        indicator_layout.addWidget(self.thinking_indicator)

        self.thinking_label.hide()
        self.thinking_indicator.hide()
        layout.addLayout(indicator_layout)

        hbox1_layout = QHBoxLayout()

        self.database_pulldown = RefreshingComboBox(self)
        self.database_pulldown.setToolTip(TOOLTIPS["DATABASE_SELECT"])
        self.database_pulldown.addItems(self.load_created_databases())
        hbox1_layout.addWidget(self.database_pulldown)

        self.model_source_combo = QComboBox()
        self.model_source_combo.setToolTip(TOOLTIPS["MODEL_BACKEND_SELECT"])
        self.model_source_combo.addItems([
            "Local Model",
            "Kobold",
            "LM Studio",
            *OPENAI_MODELS,
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
        ])

        self.model_source_combo.setCurrentText("Local Model")
        self.model_source_combo.currentTextChanged.connect(self.on_model_source_changed)
        hbox1_layout.addWidget(self.model_source_combo)

        self.model_combo_box = QComboBox()
        self.model_combo_box.setToolTip(TOOLTIPS["LOCAL_MODEL_SELECT"])
        if torch.cuda.is_available():
            for model_info in CHAT_MODELS.values():
                idx = self.model_combo_box.count()
                self.model_combo_box.addItem(model_info["model"])
                gb = round(model_info["vram"] / 1024, 1)
                self.model_combo_box.setItemData(idx, f"Uses ~{gb} GB memory", Qt.ToolTipRole)
            self.model_combo_box.setEnabled(True)
        else:
            for key in [
                "LiquidAI - .35b",
                "Qwen 3 - 0.6b (Thinking)",
                "LiquidAI - 1.2b",
                "Qwen 3 - 1.7b (Thinking)",
                "Granite - 2b",
            ]:
                self.model_combo_box.addItem(CHAT_MODELS[key]["model"])
            self.model_combo_box.setToolTip("Choose a local model. It will be downloaded.")
        if self.model_combo_box.count() > 0:
            self.model_combo_box.setCurrentIndex(0)
        hbox1_layout.addWidget(self.model_combo_box)

        self.eject_button = QPushButton("Eject Local Model")
        self.eject_button.setToolTip(TOOLTIPS["EJECT_LOCAL_MODEL"])
        self.eject_button.clicked.connect(self.eject_model)
        self.eject_button.setEnabled(False)
        hbox1_layout.addWidget(self.eject_button)

        if not torch.cuda.is_available():
            self.model_source_combo.setItemData(0, 0, Qt.UserRole - 1)
            tooltip = "The Local Model option requires GPU-acceleration."
            self.model_source_combo.setItemData(0, tooltip, Qt.ToolTipRole)
            self.model_combo_box.setEnabled(False)
            self.model_combo_box.setToolTip(tooltip)
            self.model_combo_box.setStyleSheet("QComboBox:disabled { color: #707070; }")

        layout.addLayout(hbox1_layout)

        self.text_input = QTextEdit()
        self.text_input.setToolTip(TOOLTIPS["QUESTION_INPUT"])
        self.text_input.setMaximumHeight(80)
        layout.addWidget(self.text_input, 1)

        toggles_row = QHBoxLayout()

        self.show_thinking_checkbox = QCheckBox("Show Thinking")
        self.show_thinking_checkbox.setChecked(False)
        self.show_thinking_checkbox.stateChanged.connect(self.toggle_thinking_visibility)
        toggles_row.addWidget(self.show_thinking_checkbox)

        self.chunks_only_checkbox = QCheckBox("Chunks Only")
        self.chunks_only_checkbox.setToolTip(TOOLTIPS["CHUNKS_ONLY"])
        toggles_row.addWidget(self.chunks_only_checkbox)

        toggles_row.addStretch(1)

        layout.addLayout(toggles_row)

        actions_row = QHBoxLayout()

        self.copy_response_button = QPushButton("Copy Response")
        self.copy_response_button.setToolTip(TOOLTIPS["COPY_RESPONSE"])
        self.copy_response_button.clicked.connect(self.on_copy_response_clicked)
        actions_row.addWidget(self.copy_response_button)

        self.bark_button = QPushButton("Speak Response")
        self.bark_button.setToolTip(TOOLTIPS["SPEAK_RESPONSE"])
        self.bark_button.clicked.connect(self.on_bark_button_clicked)
        actions_row.addWidget(self.bark_button)

        self.record_button = QPushButton("Voice Recorder")
        self.record_button.setToolTip(TOOLTIPS["VOICE_RECORDER"])
        self.record_button.clicked.connect(self.toggle_recording)
        actions_row.addWidget(self.record_button)

        self.submit_button = QPushButton("Submit Question")
        self.submit_button.clicked.connect(self.on_submit_button_clicked)
        self.submit_button.setStyleSheet(CustomButtonStyles.GREEN_BUTTON_STYLE)
        self.submit_button.setDefault(True)
        actions_row.addWidget(self.submit_button)

        layout.addLayout(actions_row)

        self.is_recording = False
        self.voice_recorder = VoiceRecorder(self)

    def _strategy_for_source(self, source: str) -> SubmitStrategy:
        STRATEGIES = {
            "Local Model": LocalModelStrategy(self),
            "LM Studio": LMStudioStrategy(self),
            "Kobold": KoboldStrategy(self),
            "MiniMax-M2.7": MiniMaxStrategy(self),
            "MiniMax-M2.7-highspeed": MiniMaxStrategy(self),
        }
        for openai_model in OPENAI_MODELS:
            STRATEGIES[openai_model] = ChatGPTStrategy(self)
        try:
            return STRATEGIES[source]
        except KeyError:
            raise ValueError(f"Unknown model source: {source}")

    def setup_signals(self):
        self.local_model_chat.signals.response_signal.connect(self.update_response_local_model)
        self.local_model_chat.signals.citations_signal.connect(self.display_citations_in_widget)
        self.local_model_chat.signals.error_signal.connect(self.show_error_message)
        self.local_model_chat.signals.finished_signal.connect(self.on_submission_finished)
        self.local_model_chat.signals.model_loaded_signal.connect(self.on_model_loaded)
        self.local_model_chat.signals.model_unloaded_signal.connect(self.on_model_unloaded)
        self.local_model_chat.signals.token_count_signal.connect(self.update_token_count_label)

    def _render_html(self):
        if self.show_thinking_checkbox.isChecked():
            visible_text = self.raw_response
        else:
            txt = self.raw_response
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.IGNORECASE)
            txt = re.sub(r"<think>.*$", "", txt, flags=re.DOTALL | re.IGNORECASE)
            txt = re.sub(r"\n\s*\n", "\n", txt).lstrip()
            visible_text = txt

        body = html.escape(visible_text).replace("\n", "<br>")
        body += self.citations_block

        self.response_widget.setHtml(body)
        self.response_widget.verticalScrollBar().setValue(
            self.response_widget.verticalScrollBar().maximum())

    def toggle_thinking_visibility(self):
        self._render_html()

    def update_token_count_label(self, token_count_string):
        self.token_count_label.setText(token_count_string)

    def on_model_source_changed(self, text):
        is_local = text == "Local Model"
        self.model_combo_box.setVisible(is_local)
        self.eject_button.setVisible(is_local)
        if is_local:
            self.model_combo_box.setEnabled(torch.cuda.is_available())
            self.eject_button.setEnabled(self.local_model_chat.is_model_loaded())
        else:
            self.model_combo_box.setEnabled(False)
            self.eject_button.setEnabled(False)

    def load_created_databases(self):
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                databases = list(config.get('created_databases', {}).keys())
                return [db for db in databases if db != "user_manual"]
        return []

    def on_submit_button_clicked(self):
        script_dir = PROJECT_ROOT
        selected_database = self.database_pulldown.currentText()
        if not selected_database or not (script_dir / "Vector_DB" / selected_database).exists():
            QMessageBox.warning(self, "No Database Selected", "Select a vector database to query first.")
            return

        self.response_widget.clear()
        self.token_count_label.clear()
        cursor = self.response_widget.textCursor()
        cursor.clearSelection()
        self.response_widget.setTextCursor(cursor)

        self.raw_response = ""
        self.citations_block = ""
        self.submit_button.setDisabled(True)
        user_question = self.text_input.toPlainText()

        if self.chunks_only_checkbox.isChecked():
            strategy = ChunksOnlyStrategy(self)
        else:
            strategy = self._strategy_for_source(self.model_source_combo.currentText())

        try:
            strategy.submit(user_question, selected_database)
        except Exception as e:
            logging.exception("Submission failed: %s", e)
            self.show_error_message(str(e))
            self.submit_button.setDisabled(False)

    def display_chunks(self, chunks):
        self.response_widget.setPlainText(chunks)

    def on_database_query_finished(self):
        self.submit_button.setDisabled(False)

    def eject_model(self):
        if self.local_model_chat.is_model_loaded():
            try:
                self.local_model_chat.eject_model()
            except Exception as e:
                logging.exception(f"Error during model ejection: {e}")
            finally:
                self.eject_button.setEnabled(False)
                self.model_combo_box.setEnabled(True)
        else:
            logging.warning("No model is currently loaded.")

    def on_model_loaded(self):
        self.eject_button.setEnabled(True)
        self.eject_button.setText(f"Eject {self.local_model_chat.current_model}")

    def on_model_unloaded(self):
        self.eject_button.setEnabled(False)
        self.eject_button.setText("Eject Local Model")

    def display_citations_in_widget(self, citations):
        if citations:
            self.citations_block = f"<br><br>Citation Links:{citations}"
        else:
            self.citations_block = "<br><br>No citations found."
        self._render_html()

    def on_copy_response_clicked(self):
        clipboard = QApplication.clipboard()
        response_text = self.response_widget.toPlainText()
        if response_text:
            clipboard.setText(response_text)
            QMessageBox.information(self, "Information", "Response copied to clipboard.")
        else:
            QMessageBox.warning(self, "Warning", "No response to copy.")

    def on_bark_button_clicked(self):
        script_dir = PROJECT_ROOT
        config_path = script_dir / 'config.yaml'

        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
            tts_config = config.get('tts', {})

        tts_model = tts_config.get('model', '').lower()

        if tts_model not in ['googletts', 'chattts', 'kyutaipocket', 'chatterbox'] and not torch.cuda.is_available():
            QMessageBox.warning(self, "Error", "The Text to Speech backend you selected requires GPU-acceleration.")
            return

        from core.utilities import check_backend_dependencies, install_packages
        from core.constants import BACKEND_DEPENDENCIES
        
        if not check_backend_dependencies(tts_model, interactive=False):
            required_packages = BACKEND_DEPENDENCIES.get(tts_model, {})
            if required_packages:
                packages_str = ", ".join([f"{pkg}=={ver}" for pkg, ver in required_packages.items()])

                reply = QMessageBox.question(
                    self, 
                    "Missing Dependencies",
                    f"{tts_model.title()} backend requires additional packages:\n\n{packages_str}\n\nInstall now?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    missing_packages = [(pkg, ver) for pkg, ver in required_packages.items()]
                    if install_packages(missing_packages):
                        QMessageBox.information(self, "Success", "Dependencies installed successfully!")
                    else:
                        QMessageBox.warning(self, "Installation Failed", "Failed to install dependencies. Please install manually.")
                        return
                else:
                    return

        if not (script_dir / 'chat_history.txt').exists():
            QMessageBox.warning(self, "Error", "No response to play.")
            return

        self.run_tts_module()

    def run_tts_module(self):
        process = multiprocessing.Process(target=run_tts_in_process, args=(str(self.config_path), input_text_file))
        process.start()

    def toggle_recording(self):
        if self.is_recording:
            self.voice_recorder.stop_recording()
            self.record_button.setText("Voice Recorder")
        else:
            self.voice_recorder.start_recording()
            self.record_button.setText("Stop Recording")
        self.is_recording = not self.is_recording

    def update_response_lm_studio(self, response_chunk):
        self.raw_response += response_chunk
        self._render_html()
        self.response_widget.verticalScrollBar().setValue(
            self.response_widget.verticalScrollBar().maximum()
        )

    def update_response_local_model(self, chunk: str):
        chunk_lower = chunk.lower()
        open_pos = chunk_lower.rfind("<think>")
        close_pos = chunk_lower.rfind("</think>")
        
        if open_pos != -1 or close_pos != -1:
            self.in_think_block = open_pos > close_pos

        visible = self.in_think_block and not self.show_thinking_checkbox.isChecked()
        self.thinking_indicator.setVisible(visible)
        self.thinking_label.setVisible(visible)

        self.raw_response += chunk
        self._render_html()

    def show_error_message(self, error_message):
        if "exceed the chat model's context limit" in error_message:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(error_message)
            msg_box.setWindowTitle("Context Limit Exceeded")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
        else:
            QMessageBox.warning(self, "Error", error_message)
        self.submit_button.setDisabled(False)

    def on_submission_finished(self):
        self.submit_button.setDisabled(False)

        ix = self.raw_response.lower().rfind("</think>")
        answer_only = self.raw_response[ix + len("</think>"):] if ix != -1 else self.raw_response
        answer_only = answer_only.lstrip("\n")

        try:
            with open(input_text_file, "w", encoding="utf-8") as f:
                f.write(normalize_chat_text(answer_only))
        except OSError as e:
            logging.exception(f"Could not write chat_history.txt: {e}")

    def update_transcription(self, transcription_text):
        self.text_input.setPlainText(transcription_text)

    def cleanup(self):
        if self.local_model_chat.is_model_loaded():
            self.local_model_chat.eject_model()
        if self.database_query_thread and self.database_query_thread.isRunning():
            self.database_query_thread.stop()
            self.database_query_thread.wait()
        if self.chatgpt_thread and self.chatgpt_thread.isRunning():
            self.chatgpt_thread.wait()
        if self.minimax_thread and self.minimax_thread.isRunning():
            self.minimax_thread.wait()
        if self.kobold_thread and self.kobold_thread.isRunning():
            self.kobold_thread.stop()
            self.kobold_thread.wait(timeout=5000)
        print("Cleanup completed")
