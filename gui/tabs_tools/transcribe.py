import threading
from pathlib import Path
import yaml
import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QSlider
)
from module_transcribe import WhisperTranscriber
from utilities import my_cprint, has_bfloat16_support
from constants import WHISPER_MODELS, TOOLTIPS

class TranscriberToolSettingsTab(QWidget):
    CONFIG_FILE = 'config.yaml'
    
    def __init__(self):
        super().__init__()
        self.selected_audio_file = None
        self.create_layout()

    def set_buttons_enabled(self, enabled):
        self.transcribe_button.setEnabled(enabled)
        self.select_file_button.setEnabled(enabled)

    def create_layout(self):
        main_layout = QVBoxLayout()
        
        model_selection_hbox = QHBoxLayout()
        model_label = QLabel("Model")
        model_label.setToolTip(TOOLTIPS["WHISPER_MODEL_SELECT"])
        model_selection_hbox.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.populate_model_combo()
        self.model_combo.setToolTip(TOOLTIPS["WHISPER_MODEL_SELECT"])
        model_selection_hbox.addWidget(self.model_combo)
        
        batch_label = QLabel("Batch:")
        batch_label.setToolTip(TOOLTIPS["WHISPER_BATCH_SIZE"])
        model_selection_hbox.addWidget(batch_label)
        
        self.slider_label = QLabel("8")
        self.slider_label.setToolTip(TOOLTIPS["WHISPER_BATCH_SIZE"])
        
        self.number_slider = QSlider(Qt.Horizontal)
        self.number_slider.setMinimum(1)
        self.number_slider.setMaximum(150)
        self.number_slider.setValue(8)
        self.number_slider.valueChanged.connect(self.update_slider_label)
        self.number_slider.setToolTip(TOOLTIPS["WHISPER_BATCH_SIZE"])
        
        model_selection_hbox.addWidget(self.number_slider)
        model_selection_hbox.addWidget(self.slider_label)
        
        model_selection_hbox.setStretchFactor(self.model_combo, 2)
        model_selection_hbox.setStretchFactor(self.number_slider, 2)
        
        main_layout.addLayout(model_selection_hbox)
        
        hbox = QHBoxLayout()
        self.select_file_button = QPushButton("Select Audio File")
        self.select_file_button.clicked.connect(self.select_audio_file)
        self.select_file_button.setToolTip(TOOLTIPS["AUDIO_FILE_SELECT"])
        hbox.addWidget(self.select_file_button)
        
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.transcribe_button.setToolTip(TOOLTIPS["TRANSCRIBE_BUTTON"])
        hbox.addWidget(self.transcribe_button)
        
        main_layout.addLayout(hbox)
        
        self.file_path_label = QLabel("No file currently selected")
        main_layout.addWidget(self.file_path_label)
        
        self.setLayout(main_layout)

    def populate_model_combo(self):
        cuda_available = torch.cuda.is_available()
        bfloat16_supported = has_bfloat16_support()

        filtered_models = []
        for model_name, model_info in WHISPER_MODELS.items():
            precision = model_info['precision']
            if precision == 'float32':
                filtered_models.append(model_name)
            elif precision == 'bfloat16' and bfloat16_supported:
                filtered_models.append(model_name)
            elif precision == 'float16' and cuda_available:
                filtered_models.append(model_name)

        self.model_combo.addItems(filtered_models)

    def update_slider_label(self, value):
        self.slider_label.setText(str(value))

    def update_config_file(self):
        with open(self.CONFIG_FILE, 'w') as file:
            yaml.dump(self.config, file)

    def select_audio_file(self):
        current_dir = Path.cwd()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Audio File", str(current_dir))
        if file_name:
            file_path = Path(file_name)
            short_path = f"...{file_path.parent.name}/{file_path.name}"
            self.file_path_label.setText(short_path)
            self.file_path_label.setToolTip(str(file_path.absolute()))
            self.selected_audio_file = file_name

    def start_transcription(self):
        if not self.selected_audio_file:
            print("Please select an audio file.")
            return
        
        selected_model_key = self.model_combo.currentText()
        selected_batch_size = int(self.slider_label.text())
        
        def transcription_thread():
            self.set_buttons_enabled(False)
            try:
                transcriber = WhisperTranscriber(
                    model_key=selected_model_key, 
                    batch_size=selected_batch_size
                )
                transcriber.start_transcription_process(self.selected_audio_file)
                my_cprint("Transcription created and ready to be input into vector database.", 'green')
            finally:
                self.set_buttons_enabled(True)
        
        threading.Thread(target=transcription_thread, daemon=True).start()