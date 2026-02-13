import yaml
from pathlib import Path
import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QGridLayout, QVBoxLayout, QComboBox, QWidget
from core.constants import VISION_MODELS

CONFIG_FILE = "config.yaml"


def _read_cfg() -> dict:
    p = Path(CONFIG_FILE)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _write_cfg(cfg: dict) -> None:
    with Path(CONFIG_FILE).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)


def is_cuda_available():
    return torch.cuda.is_available()


def get_cuda_capability():
    if is_cuda_available():
        return torch.cuda.get_device_capability(0)
    return (0, 0)


class VisionSettingsTab(QWidget):

    def __init__(self):
        super().__init__()
        mainVLayout = QVBoxLayout()
        self.setLayout(mainVLayout)

        gridLayout = QGridLayout()
        mainVLayout.addLayout(gridLayout)

        label_model = QLabel("Model")
        gridLayout.addWidget(label_model, 0, 0, Qt.AlignCenter)

        label_size = QLabel("Size")
        gridLayout.addWidget(label_size, 0, 1, Qt.AlignCenter)

        label_vram = QLabel("VRAM")
        gridLayout.addWidget(label_vram, 0, 2, Qt.AlignCenter)

        label_vision = QLabel("Vision Component")
        gridLayout.addWidget(label_vision, 0, 3, Qt.AlignCenter)

        label_chat = QLabel("Chat Component")
        gridLayout.addWidget(label_chat, 0, 4, Qt.AlignCenter)

        label_avg = QLabel("Avg Length")
        gridLayout.addWidget(label_avg, 0, 5, Qt.AlignCenter)

        self.modelComboBox = QComboBox()
        self.populate_model_combobox()
        self.modelComboBox.setMinimumWidth(175)
        gridLayout.addWidget(self.modelComboBox, 1, 0, Qt.AlignCenter)

        self.sizeLabel = QLabel("—")
        gridLayout.addWidget(self.sizeLabel, 1, 1, Qt.AlignCenter)

        self.vramLabel = QLabel("—")
        gridLayout.addWidget(self.vramLabel, 1, 2, Qt.AlignCenter)

        self.visionComponentLabel = QLabel("—")
        self.visionComponentLabel.setWordWrap(True)
        gridLayout.addWidget(self.visionComponentLabel, 1, 3, Qt.AlignCenter)

        self.chatComponentLabel = QLabel("—")
        self.chatComponentLabel.setWordWrap(True)
        gridLayout.addWidget(self.chatComponentLabel, 1, 4, Qt.AlignCenter)

        self.avgLenLabel = QLabel("—")
        gridLayout.addWidget(self.avgLenLabel, 1, 5, Qt.AlignCenter)

        cfg = _read_cfg()
        saved = (cfg.get("vision") or {}).get("chosen_model")
        if saved and saved in VISION_MODELS:
            self.modelComboBox.setCurrentText(saved)

        self.modelComboBox.currentTextChanged.connect(self._apply_model_to_labels)

        self._apply_model_to_labels(self.modelComboBox.currentText())

    def populate_model_combobox(self):
        self.modelComboBox.clear()
        self.modelComboBox.addItems(VISION_MODELS.keys())

    def _apply_model_to_labels(self, model_name: str):
        info = VISION_MODELS.get(model_name, {}) or {}

        size = info.get("size", "—")
        vram = info.get("vram", "—")
        vision_component = info.get("vision_component", "—")
        chat_component = info.get("chat_component", "—")
        avg_length = info.get("avg_length", "—")

        self.sizeLabel.setText(str(size))
        self.vramLabel.setText(str(vram))
        self.visionComponentLabel.setText(str(vision_component))
        self.chatComponentLabel.setText(str(chat_component))
        self.avgLenLabel.setText(str(avg_length))

        cfg = _read_cfg()
        cfg.setdefault("vision", {})
        if cfg["vision"].get("chosen_model") != model_name:
            cfg["vision"]["chosen_model"] = model_name
            _write_cfg(cfg)
