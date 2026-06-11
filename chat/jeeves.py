import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path

from core.utilities import set_cuda_paths
set_cuda_paths()

import yaml
from core.utilities import ensure_theme_config, load_stylesheet

from ctypes import windll, byref, sizeof, c_int
from ctypes.wintypes import BOOL, HWND, DWORD
import gc
import torch
import re
import time
import random
import chat.base as module_chat
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTextEdit, QTextBrowser,
    QLineEdit, QMessageBox, QPushButton, QLabel,
    QHBoxLayout, QComboBox, QApplication
)
from PySide6.QtCore import QThread, Signal, Qt, QObject, QUrl
from PySide6.QtGui import QTextCursor, QPixmap, QDesktopServices
from core.constants import (
    CustomButtonStyles,
    JEEVES_MODELS,
    PROJECT_ROOT,
)
from db.database_interactions import get_query_db
from modules.kokoro import KokoroTTS
from core.utilities import normalize_chat_text


JEEVES_RAG_INSTRUCTION = (
    "The excerpts below are from this program's user guide. Answer the user's question using them "
    "as your source. Give a direct, specific, helpful answer; if the excerpts only partially cover "
    "the question, still give the most useful answer you can from them. Never tell the user that the "
    "excerpts do not address the question; simply answer as best you can, in character as the butler."
)


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


class TextStreamWorker(QThread):
    chunk_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, text, delay=0.018):
        super().__init__()
        self.text = text
        self.delay = delay
        self._is_running = True

    def run(self):
        for token in re.findall(r'\S+|\s+', self.text):
            if not self._is_running:
                break
            self.chunk_signal.emit(token)
            if token.strip():
                time.sleep(self.delay)
        self.finished_signal.emit()

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
            model_instance = load_jeeves_model(self.chat_model_key)
            self.loaded_signal.emit(model_instance)
        except Exception as e:
            self.error_signal.emit(str(e))

class ChatWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ask Jeeves (Welcome back Jeeves!)")
        self.setGeometry(100, 100, 850, 950)

        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(1)

        image_path = PROJECT_ROOT / "Assets" / "ask_jeeves_transparent.jpg"
        if image_path.exists():
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                image_label = QLabel()
                image_label.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image_label.setAlignment(Qt.AlignCenter)
                self.layout.addWidget(image_label)

        model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.setFixedHeight(30)
        self.model_selector.addItem("Please choose a model...")

        self.model_selector.addItems(JEEVES_MODELS)
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)
        model_layout.addWidget(self.model_selector)

        self.eject_button = QPushButton("Eject")
        self.eject_button.setFixedHeight(30)
        self.eject_button.clicked.connect(self.eject_model)
        self.eject_button.setEnabled(False)
        model_layout.addWidget(self.eject_button)

        self.layout.addLayout(model_layout)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlainText("Hello, my name is Jeeves. Thank you for the job opportunity! Ask me how to use this program.")
        self.layout.addWidget(self.chat_display, 4)

        self.sources_toggle = QPushButton("Show sources")
        self.sources_toggle.setCheckable(True)
        self.sources_toggle.setFixedHeight(26)
        self.sources_toggle.setStyleSheet("text-align: left; padding: 1px 10px;")
        self.sources_toggle.setVisible(False)
        self.sources_toggle.toggled.connect(self._toggle_sources)
        self.layout.addWidget(self.sources_toggle)

        self.sources_view = QTextBrowser()
        self.sources_view.setOpenLinks(False)
        self.sources_view.setMaximumHeight(220)
        self.sources_view.setVisible(False)
        self.sources_view.anchorClicked.connect(self._open_source_link)
        self.layout.addWidget(self.sources_view)

        input_row_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setFixedHeight(30)
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        input_row_layout.addWidget(self.input_field, stretch=4)

        self.speak_button = QPushButton("Speak Response")
        self.speak_button.setEnabled(False)
        self.speak_button.setFixedHeight(30)
        self.speak_button.clicked.connect(self.toggle_speech)
        self.speak_button.setStyleSheet(CustomButtonStyles.TEAL_BUTTON_STYLE)
        input_row_layout.addWidget(self.speak_button)

        self.voice_select = QComboBox()
        self.voice_select.setEnabled(False)
        self.voice_select.addItems(['bm_george', 'bm_lewis', 'bf_isabella', 'af'])
        self.voice_select.setCurrentText('bm_george')
        self.voice_select.setFixedHeight(30)
        input_row_layout.addWidget(self.voice_select)

        self.speed_control = QComboBox()
        self.speed_control.setEnabled(False)
        self.speed_mapping = {
            'Slow': 1.0,
            'Medium': 1.3,
            'Fast': 1.6
        }
        self.speed_control.addItems(list(self.speed_mapping.keys()))
        self.speed_control.setCurrentText('Medium')
        self.speed_control.setFixedHeight(30)
        input_row_layout.addWidget(self.speed_control)

        self.layout.addLayout(input_row_layout)

        self.poem_panel = QWidget()
        self.poem_layout = QHBoxLayout(self.poem_panel)
        self.poem_layout.setContentsMargins(0, 0, 0, 0)
        self.poem_panel.setVisible(False)
        self.layout.addWidget(self.poem_panel)

        self.setCentralWidget(central_widget)

        self.model_instance = None
        self.worker = None
        self._load_worker = None
        self.last_contexts = []
        self.last_metadata = []
        self._current_response = ""
        self.poems = self._load_poems()
        self._text_worker = None
        self.poem_combo = None

        self.vector_db = get_query_db("user_manual")

        try:
            tts_path = PROJECT_ROOT / "Models" / "tts" / "ctranslate2-4you--Kokoro-82M-light"
            self.tts = KokoroTTS(repo_path=str(tts_path))
            self.speak_button.setEnabled(True)
            self.voice_select.setEnabled(True)
            self.speed_control.setEnabled(True)
        except Exception:
            self.tts = None

        self.tts_thread = None
        self.tts_worker = None
        self.is_speaking = False

    def eject_model(self):
        if self.model_instance:
            try:
                self.model_instance.cleanup()
            except Exception:
                pass
            self.model_instance = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model_selector.setCurrentIndex(0)
        self.eject_button.setEnabled(False)
        gc.collect()

    def toggle_speech(self):
        if self.is_speaking:
            self.cancel_speech()
        else:
            self.speak_response()

    def on_model_selected(self, index):
        if index == 0:
            if self.model_instance:
                self.eject_model()
            return

        model_key = self.model_selector.currentText()

        # Free any currently-loaded model before loading the new one (without resetting the selector).
        if self.model_instance:
            try:
                self.model_instance.cleanup()
            except Exception:
                pass
            self.model_instance = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.model_selector.setEnabled(False)
        self.input_field.setEnabled(False)
        self.eject_button.setEnabled(False)
        self.chat_display.setPlainText(
            f"Loading {model_key} ... (the first time, this downloads the model -- please wait.)"
        )

        self._load_worker = ModelLoadWorker(model_key)
        self._load_worker.loaded_signal.connect(self._on_model_loaded)
        self._load_worker.error_signal.connect(self._on_load_error)
        self._load_worker.start()

    def _on_model_loaded(self, model_instance):
        self.model_instance = model_instance
        self.model_selector.setEnabled(True)
        self.input_field.setEnabled(True)
        self.eject_button.setEnabled(True)
        self.chat_display.setPlainText(
            "Hello, my name is Jeeves. Thank you for the job opportunity! Ask me how to use this program."
        )
        if self._load_worker:
            self._load_worker.quit()
            self._load_worker.wait()
            self._load_worker = None

    def _on_load_error(self, error_message):
        self.model_selector.setEnabled(True)
        self.input_field.setEnabled(True)
        self.eject_button.setEnabled(False)
        self.model_selector.blockSignals(True)
        self.model_selector.setCurrentIndex(0)
        self.model_selector.blockSignals(False)
        QMessageBox.warning(self, "Model Load Error", f"Could not load the model:\n{error_message}")
        if self._load_worker:
            self._load_worker.quit()
            self._load_worker.wait()
            self._load_worker = None

    def showEvent(self, event):
        super().showEvent(event)
        self.apply_dark_mode_settings()

    def apply_dark_mode_settings(self):
        DWMWA_USE_IMMERSIVE_DARK_MODE = DWORD(20)
        set_window_attribute = windll.dwmapi.DwmSetWindowAttribute
        hwnd = HWND(int(self.winId()))
        true_bool = BOOL(True)
        set_window_attribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            byref(true_bool),
            sizeof(true_bool)
        )

        DWMWA_BORDER_COLOR = DWORD(34)
        black_color = c_int(0xFF000000)
        set_window_attribute(
            hwnd,
            DWMWA_BORDER_COLOR,
            byref(black_color),
            sizeof(black_color)
        )

    def send_message(self):
        if not self.model_instance:
            QMessageBox.warning(self, "No Model Selected",
                              "Please select a language model before sending a message.")
            return

        if self.worker and self.worker.isRunning():
            return

        user_message = self.input_field.text().strip()
        if not user_message:
            return

        self.chat_display.clear()
        self.sources_toggle.setVisible(False)
        self.sources_view.setVisible(False)

        try:
            contexts, metadata = self.vector_db.search(user_message, k=5, score_threshold=0.5)
            if not contexts:
                QMessageBox.warning(
                    self, "No Contexts Found",
                    "No relevant chunks were found in the user manual database for this question. "
                    "Try rephrasing your question."
                )
                return
        except Exception as e:
            QMessageBox.warning(self, "Database Query Error", f"An error occurred while querying the database: {e}")
            return

        self.last_contexts = contexts
        self.last_metadata = metadata
        self._current_response = ""

        contexts_text = "\n\n".join(contexts)
        augmented_query = (
            f"{JEEVES_RAG_INSTRUCTION}\n\n"
            f"USER GUIDE EXCERPTS:\n{contexts_text}\n\n"
            f"QUESTION: {user_message}"
        )

        self.input_field.clear()
        self.input_field.setDisabled(True)
        self.chat_display.append(f"User: {user_message}")
        self.chat_display.append("\nJeeves: ")

        self.worker = GenerationWorker(self.model_instance, augmented_query)
        self.worker.token_signal.connect(self.update_response)
        self.worker.error_signal.connect(self.show_error)
        self.worker.finished_signal.connect(self.on_generation_finished)
        self.worker.start()

    def update_response(self, token):
        self._current_response += token
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText(token)
        self.chat_display.ensureCursorVisible()

    def show_error(self, error_message):
        QMessageBox.warning(self, "Error", f"An error occurred: {error_message}")
        self.input_field.setDisabled(False)

    def on_generation_finished(self):
        self._populate_sources()
        if self.worker:
            if self.worker.isRunning():
                self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
        if self.poems and random.random() < (1.0 / 3.0):
            self._offer_poem()
        else:
            self.input_field.setDisabled(False)
            self.input_field.setFocus()

    def _load_poems(self):
        poems = []
        path = PROJECT_ROOT / "Assets" / "jeeves_poems.txt"
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            for block in raw.split("@@@@@"):
                block = block.strip("\n").rstrip()
                if not block.strip():
                    continue
                plines = block.split("\n")
                title = plines[0].strip().strip('"' + chr(0x201c) + chr(0x201d))
                author = ""
                for ln in plines[1:]:
                    if ln.strip():
                        author = re.sub(r'^(?:[Bb]y[:\s]+)', '', ln.strip()).strip()
                        break
                label = f"{title} - {author}" if author else title
                poems.append({"title": title, "label": label, "text": block})
        return poems

    def _append_chunk(self, chunk):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText(chunk)
        self.chat_display.ensureCursorVisible()

    def _stream_text(self, text, on_done):
        self._text_worker = TextStreamWorker(text)
        self._text_worker.chunk_signal.connect(self._append_chunk)
        self._text_worker.finished_signal.connect(on_done)
        self._text_worker.start()

    def _clear_poem_panel(self):
        while self.poem_layout.count():
            item = self.poem_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.poem_combo = None

    def _offer_poem(self):
        self.input_field.setDisabled(True)
        self._append_chunk("\n\nJeeves: ")
        offer = random.choice([
            "If I may be so bold, sir or madam -- might I interest you in a poem to pass the time?",
            "Pardon the intrusion, but might I recite a poem for your enjoyment?",
            "If you can spare a moment, perhaps a spot of poetry would be agreeable?",
        ])
        self._stream_text(offer, self._show_poem_yesno)

    def _show_poem_yesno(self):
        self._clear_poem_panel()
        yes_btn = QPushButton("Yes, please")
        yes_btn.setStyleSheet(CustomButtonStyles.TEAL_BUTTON_STYLE)
        yes_btn.clicked.connect(self._poem_yes)
        no_btn = QPushButton("No, thank you")
        no_btn.clicked.connect(self._poem_no)
        self.poem_layout.addWidget(yes_btn)
        self.poem_layout.addWidget(no_btn)
        self.poem_panel.setVisible(True)

    def _poem_no(self):
        self._clear_poem_panel()
        self.poem_panel.setVisible(False)
        self._append_chunk("\n\nJeeves: Very good, sir. Perhaps another time.")
        self._end_poem_mode()

    def _poem_yes(self):
        self._clear_poem_panel()
        self.poem_panel.setVisible(False)
        self.chat_display.clear()
        self._append_chunk("Jeeves: ")
        self._stream_text(
            "Splendid! Which poem shall I recite for you? Kindly make your selection below.",
            self._show_poem_choices,
        )

    def _show_poem_choices(self):
        self._clear_poem_panel()
        self.poem_combo = QComboBox()
        self.poem_combo.addItems([p["label"] for p in self.poems])
        recite_btn = QPushButton("Recite")
        recite_btn.setStyleSheet(CustomButtonStyles.TEAL_BUTTON_STYLE)
        recite_btn.clicked.connect(self._poem_recite)
        cancel_btn = QPushButton("Never mind")
        cancel_btn.clicked.connect(self._poem_cancel)
        self.poem_layout.addWidget(self.poem_combo, 1)
        self.poem_layout.addWidget(recite_btn)
        self.poem_layout.addWidget(cancel_btn)
        self.poem_panel.setVisible(True)

    def _poem_cancel(self):
        self._clear_poem_panel()
        self.poem_panel.setVisible(False)
        self._append_chunk("\n\nJeeves: Very good, sir.")
        self._end_poem_mode()

    def _poem_recite(self):
        idx = self.poem_combo.currentIndex() if self.poem_combo else -1
        self._clear_poem_panel()
        self.poem_panel.setVisible(False)
        if idx < 0 or idx >= len(self.poems):
            self._end_poem_mode()
            return
        poem = self.poems[idx]
        self.chat_display.clear()
        self._append_chunk("Jeeves:\n\n")
        self._stream_text(poem["text"], self._end_poem_mode)

    def _end_poem_mode(self):
        self.input_field.setDisabled(False)
        self.input_field.setFocus()

    def _populate_sources(self):
        if not self.last_contexts:
            self.sources_toggle.setVisible(False)
            self.sources_view.setVisible(False)
            return

        def esc(s):
            return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        parts = ['<div style="font-size:9pt;">']
        for i, chunk in enumerate(self.last_contexts):
            meta = self.last_metadata[i] if i < len(self.last_metadata) else {}
            fp = str(meta.get('file_path') or '')
            src = meta.get('file_name') or os.path.basename(fp) or 'unknown source'
            if fp.startswith(('http://', 'https://')):
                href = fp
            elif fp:
                href = QUrl.fromLocalFile(fp).toString()
            else:
                href = ''
            if href:
                name_html = f'<a href="{esc(href)}" style="color:#5aa0e0; text-decoration:none;">{esc(src)}</a>'
            else:
                name_html = f'<b>{esc(src)}</b>'
            score = meta.get('similarity_score')
            score_str = f' &nbsp;(score {score:.3f})' if isinstance(score, (int, float)) else ''
            text = (chunk or '').strip()
            if len(text) > 700:
                text = text[:700] + ' ...'
            text = esc(text).replace('\n', '<br>')
            parts.append(
                f'<p style="margin:4px 0;">{i + 1}. {name_html}{score_str}<br>'
                f'<span style="color:#9a9a9a;">{text}</span></p>'
            )
        parts.append('</div>')
        self.sources_view.setHtml(''.join(parts))

        n = len(self.last_contexts)
        self.sources_toggle.blockSignals(True)
        self.sources_toggle.setChecked(False)
        self.sources_toggle.blockSignals(False)
        self.sources_toggle.setText(f"Show sources ({n})")
        self.sources_toggle.setVisible(True)
        self.sources_view.setVisible(False)

    def _toggle_sources(self, checked):
        self.sources_view.setVisible(checked)
        n = len(self.last_contexts)
        self.sources_toggle.setText(f"{'Hide' if checked else 'Show'} sources ({n})")

    def _open_source_link(self, url):
        if url.scheme() in ('http', 'https'):
            QDesktopServices.openUrl(url)
            return
        local = url.toLocalFile() or url.path()
        if local and os.path.exists(local):
            QDesktopServices.openUrl(QUrl.fromLocalFile(local))
        else:
            QMessageBox.warning(
                self, "File Not Found",
                f"Could not open the source file:\n{local or url.toString()}"
            )

    def speak_response(self):
        if not self.tts:
            QMessageBox.warning(self, "TTS Not Available", 
                "Text-to-speech is not available. Please check if KokoroTTS is properly installed.")
            return

        selected_voice = self.voice_select.currentText()
        selected_speed = self.speed_mapping[self.speed_control.currentText()]
        
        response_text = (self._current_response or "").strip()

        if not response_text:
            QMessageBox.warning(self, "Empty Response", 
                "The response is empty. Please ask a question first.")
            return

        self.is_speaking = True
        self.speak_button.setText("Cancel Playback")
        self.voice_select.setEnabled(False)
        self.speed_control.setEnabled(False)

        self.tts_thread = QThread()

        self.tts_worker = TTSWorker(self.tts, response_text, selected_voice, selected_speed)
        self.tts_worker.moveToThread(self.tts_thread)

        self.tts_thread.started.connect(self.tts_worker.run)
        self.tts_worker.finished.connect(self.on_speech_finished)
        self.tts_worker.finished.connect(self.tts_worker.deleteLater)
        self.tts_thread.finished.connect(self.tts_thread.deleteLater)
        self.tts_worker.error.connect(self.handle_tts_error)

        self.tts_thread.start()

    def cancel_speech(self):
        if self.tts_worker:
            self.tts_worker.stop()

    def on_speech_finished(self):
        self.is_speaking = False
        self.speak_button.setText("Speak Response")
        self.speak_button.setEnabled(True)
        self.voice_select.setEnabled(True)
        self.speed_control.setEnabled(True)
        
        if self.tts_thread:
            self.tts_thread.quit()
            self.tts_thread.wait()

    def handle_tts_error(self, error_message):
        self.on_speech_finished()
        QMessageBox.warning(self, "TTS Error", 
            f"An error occurred while trying to speak: {error_message}")

    def closeEvent(self, event):
        for w in (self.worker, self._load_worker, self._text_worker):
            if w is not None and w.isRunning():
                if hasattr(w, 'stop'):
                    w.stop()
                w.wait(5000)
        if self.tts_thread is not None and self.tts_thread.isRunning():
            self.tts_thread.quit()
            self.tts_thread.wait(5000)

        if hasattr(self, 'vector_db'):
            self.vector_db.cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        event.accept()

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
            text_without_asterisks = self.text.replace('*', '')
            text_cleaned = re.sub(r'#{2,}', '', text_without_asterisks)
            normalized_text = normalize_chat_text(text_cleaned)
            
            if not self._should_stop:
                self.tts.speak(normalized_text, voice=self.voice, speed=self.speed)
            
            self.finished.emit()
        except Exception as e:
            if not self._should_stop:
                self.error.emit(str(e))

def launch_jeeves_process():
    from core.utilities import set_cuda_paths
    set_cuda_paths()
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    if hasattr(QApplication, 'setHighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication([])

    theme = ensure_theme_config()
    app.setStyleSheet(load_stylesheet(theme))

    window = ChatWindow()
    window.show()

    ret = app.exec()
    sys.exit(ret)
