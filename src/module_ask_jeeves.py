import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path

from utilities import set_cuda_paths
set_cuda_paths()

import yaml
from utilities import ensure_theme_config, load_stylesheet

from ctypes import windll, byref, sizeof, c_int
from ctypes.wintypes import BOOL, HWND, DWORD
import psutil
import ctranslate2
import gc
import torch
import re
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTextEdit, 
    QLineEdit, QMessageBox, QPushButton, QLabel,
    QHBoxLayout, QSizePolicy, QComboBox, QApplication, QSpinBox
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QObject
from PySide6.QtGui import QTextCursor, QPixmap
from constants import (
    jeeves_system_message, 
    master_questions, 
    CustomButtonStyles, 
    rag_string,
    JEEVES_MODELS
)
from download_model import ModelDownloader, model_downloaded_signal
from database_interactions import QueryVectorDB
from module_kokoro import KokoroTTS
from utilities import normalize_chat_text


class GenerationWorker(QThread):
    token_signal = Signal(str)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, generator, tokenizer, prompt, model_dir):
        super().__init__()
        self.generator = generator
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.model_dir = model_dir
        self._is_running = True

    def run(self):
        try:
            tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(self.prompt))
            try:
                endofturn_id = self.tokenizer.encode("[|endofturn|]")[0]
                use_endofturn = True
            except:
                use_endofturn = False

            model_name = Path(self.model_dir).name.lower()
            generation_params = {
                "max_length": 2048,
                "sampling_temperature": 6.0,
            }

            if "DeepSeek-R1-Distill-Qwen-1.5B" in model_name:
                generation_params["repetition_penalty"] = 1.1

            token_iterator = self.generator.generate_tokens(
                [tokens],
                **generation_params
            )

            for token_result in token_iterator:
                if not self._is_running:
                    break

                token_id = token_result.token_id
                if token_id == self.tokenizer.eos_token_id:
                    break
                if use_endofturn and token_id == endofturn_id:
                    break

                token = self.tokenizer.decode([token_id])
                self.token_signal.emit(token)

            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False

class ChatWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ask Jeeves (Welcome back Jeeves!)")
        self.setGeometry(100, 100, 850, 950)

        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(1)

        image_path = Path(__file__).parent / "Assets" / "ask_jeeves_transparent.jpg"
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

        self.model_selector.addItems(list(JEEVES_MODELS.keys()))
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)
        model_layout.addWidget(self.model_selector)

        self.eject_button = QPushButton("Eject")
        self.eject_button.setFixedHeight(30)
        self.eject_button.clicked.connect(self.eject_model)
        self.eject_button.setEnabled(False)
        model_layout.addWidget(self.eject_button)

        self.context_label = QLabel("Contexts:")
        self.context_label.setFixedHeight(30)

        self.context_spin = QSpinBox()
        self.context_spin.setRange(1, 10)
        self.context_spin.setValue(5)
        self.context_spin.setFixedHeight(30)

        model_layout.addWidget(self.context_label)
        model_layout.addWidget(self.context_spin)

        self.layout.addLayout(model_layout)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlainText("Hello, my name is Jeeves. Thank you for the job opportunity! Ask me how to use this program.")
        self.layout.addWidget(self.chat_display, 4)

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

        self.suggestion_widget = QWidget()
        self.suggestion_widget.setMinimumHeight(100)
        self.suggestion_layout = QVBoxLayout(self.suggestion_widget)
        self.suggestion_layout.setContentsMargins(0, 0, 0, 0)
        self.suggestion_layout.setSpacing(1)

        self.suggestion_buttons = []
        for _ in range(3):
            btn = QPushButton()
            btn.setVisible(True)
            btn.setStyleSheet(CustomButtonStyles.TEAL_BUTTON_STYLE)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumSize(200, 35)
            btn.clicked.connect(self.on_suggestion_clicked)
            btn.setStyleSheet("text-align: left; padding: 1px 14px;")
            self.suggestion_buttons.append(btn)
            self.suggestion_layout.addWidget(btn)

        self.suggestion_layout.addStretch()
        self.layout.addWidget(self.suggestion_widget)

        self.setCentralWidget(central_widget)

        self.model_dir = None
        self.generator = None
        self.tokenizer = None
        self.worker = None

        self.vector_db = QueryVectorDB.get_instance("user_manual")
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', token=False)
        self.question_embeddings = self.model.encode(master_questions)
        self.suggestion_cache = {}
        self.current_text = ""

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._delayed_update)

        self.input_field.textChanged.connect(self.debounce_update)

        try:
            tts_path = Path(__file__).parent / "Models" / "tts" / "ctranslate2-4you--Kokoro-82M-light"
            self.tts = KokoroTTS(repo_path=str(tts_path))
            self.speak_button.setEnabled(True)
            self.voice_select.setEnabled(True)
            self.speed_control.setEnabled(True)
        except Exception:
            self.tts = None

        self.tts_thread = None
        self.tts_worker = None
        self.is_speaking = False

    def _ensure_model(self) -> None:
        model_dir = Path(self.model_dir)
        if not (model_dir / "model.bin").exists():
            print("model.bin missing – redownloading just that file …")
            self._download_model()

    def eject_model(self):
        if self.generator:
            del self.generator
            self.generator = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
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
            if self.generator or self.tokenizer:
                self.eject_model()
            return

        model_name = self.model_selector.currentText()
        model_info = JEEVES_MODELS[model_name]
        
        self.model_dir = str(Path(__file__).parent / "Models" / "Jeeves" / model_info["folder_name"])

        if not Path(self.model_dir).exists():

            self.model_selector.setEnabled(False)
            self.input_field.setEnabled(False)
            self.eject_button.setEnabled(False)

            download_config = {
                "repo_id": model_info["repo"],
                "cache_dir": model_info["folder_name"]
            }

            self.download_worker = QThread()
            self.downloader = ModelDownloader(
                model_info=download_config,
                model_type="jeeves"
            )
            self.downloader.moveToThread(self.download_worker)

            self.download_worker.started.connect(self.downloader.download)
            model_downloaded_signal.downloaded.connect(self.on_model_downloaded)

            self.download_worker.start()
            return

        self._load_model()

    def on_model_downloaded(self, model_name, model_type):

        self.model_selector.setEnabled(True)
        self.input_field.setEnabled(True)

        self.download_worker.quit()
        self.download_worker.wait()

        self._load_model()

    def _load_model(self):
        self._ensure_model()
        physical_cores = max(1, psutil.cpu_count(logical=False) - 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.generator:
            del self.generator
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.generator = ctranslate2.Generator(
            self.model_dir,
            device=device,
            intra_threads=physical_cores,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, token=False, trust_remote_code=True)
        self.eject_button.setEnabled(True)

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

    def build_prompt(self, user_message):
        model_name = self.model_selector.currentText()
        prompt_format = JEEVES_MODELS[model_name]["prompt_format"]
        return prompt_format.format(
            jeeves_system_message=jeeves_system_message,
            user_message=user_message
        )

    def send_message(self):
        if not self.generator or not self.tokenizer:
            QMessageBox.warning(self, "No Model Selected", 
                              "Please select a language model before sending a message.")
            return

        if self.worker and self.worker.isRunning():
            return

        user_message = self.input_field.text().strip()
        if not user_message:
            return

        self.chat_display.clear()

        try:
            k_value = self.context_spin.value()
            contexts, metadata = self.vector_db.search(user_message, k=5, score_threshold=0.9)
            if not contexts:
                QMessageBox.warning(self, "No Contexts Found", "No relevant contexts were found for your query.")
                return
        except Exception as e:
            QMessageBox.warning(self, "Database Query Error", f"An error occurred while querying the database: {e}")
            return

        contexts_text = "\n\n".join(contexts)
        full_context = f"{rag_string}\n\n{contexts_text}"

        self.input_field.clear()
        self.input_field.setDisabled(True)
        self.chat_display.append(f"User: {user_message}")
        self.chat_display.append("\nAssistant: ")

        prompt = self.build_prompt(user_message)
        prompt = f"{full_context}\n\n{prompt}"

        self.worker = GenerationWorker(self.generator, self.tokenizer, prompt, self.model_dir)
        self.worker.token_signal.connect(self.update_response)
        self.worker.error_signal.connect(self.show_error)
        self.worker.finished_signal.connect(self.on_generation_finished)
        self.worker.start()

    def update_response(self, token):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertPlainText(token)
        self.chat_display.ensureCursorVisible()

    def show_error(self, error_message):
        QMessageBox.warning(self, "Error", f"An error occurred: {error_message}")
        self.input_field.setDisabled(False)

    def on_generation_finished(self):
        self.input_field.setDisabled(False)
        self.input_field.setFocus()
        if self.worker:
            if self.worker.isRunning():
                self.worker.wait()
            self.worker.deleteLater()
            self.worker = None

    def find_top_similar(self, input_text, top_k=5):
        if not input_text.strip() or len(input_text) < 3:
            return []

        input_embedding = self.model.encode([input_text])[0]
        similarities = np.dot(self.question_embeddings, input_embedding) / (
            np.linalg.norm(self.question_embeddings, axis=1) * np.linalg.norm(input_embedding)
        )

        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        threshold = 0.8
        top_questions = [
            master_questions[idx] for idx, sim in zip(top_indices, top_similarities) if sim > threshold
        ]

        return top_questions

    def debounce_update(self, text):
        self.current_text = text
        self.timer.start(500)

    def _delayed_update(self):
        text = self.current_text
        if len(text) >= 3:
            suggestions = self.find_top_similar(text, top_k=3)
            self.update_suggestions(suggestions)
        else:
            self.clear_suggestions()

    def update_suggestions(self, suggestions):
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                btn.setText(suggestions[i])
                btn.setEnabled(True)
            else:
                btn.setText("")
                btn.setEnabled(False)

    def clear_suggestions(self):
        for btn in self.suggestion_buttons:
            btn.setText("")
            btn.setEnabled(False)

    def on_suggestion_clicked(self):
        sender = self.sender()
        if sender and isinstance(sender, QPushButton):
            suggestion = sender.text()
            self.input_field.setText(suggestion)
            self.send_message()

    def speak_response(self):
        if not self.tts:
            QMessageBox.warning(self, "TTS Not Available", 
                "Text-to-speech is not available. Please check if KokoroTTS is properly installed.")
            return

        selected_voice = self.voice_select.currentText()
        selected_speed = self.speed_mapping[self.speed_control.currentText()]
        
        text = self.chat_display.toPlainText()

        try:
            response_text = text.split("Assistant: ", 1)[1].strip()
        except IndexError:
            QMessageBox.warning(self, "No Response", 
                "There is no response from Jeeves to speak. Please ask a question first.")
            return

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

    def handle_tts_error(self, error_message):
        self.speak_button.setEnabled(True)
        QMessageBox.warning(self, "TTS Error", 
            f"An error occurred while trying to speak: {error_message}")

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
    from utilities import set_cuda_paths
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