import yaml
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QGridLayout, QVBoxLayout, QComboBox, QWidget
from core.constants import VISION_MODELS
from core.utilities import save_config_atomically, runs_on_this_hardware, fallback_if_unavailable

DEFAULT_VISION_MODEL = "Liquid-VL - 480M"

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
    save_config_atomically(cfg, CONFIG_FILE, sort_keys=True)


class VisionSettingsTab(QWidget):

    def __init__(self):
        super().__init__()
        mainVLayout = QVBoxLayout()
        self.setLayout(mainVLayout)

        gridLayout = QGridLayout()
        for col, stretch in enumerate((3, 1, 1, 4, 2, 1)):
            gridLayout.setColumnStretch(col, stretch)
        mainVLayout.addLayout(gridLayout)

        for col, text in enumerate(("Model", "Size", "VRAM", "Vision Component", "Chat Component", "Avg Length")):
            header = QLabel(text)
            header.setAlignment(Qt.AlignCenter)
            gridLayout.addWidget(header, 0, col)

        self.modelComboBox = QComboBox()
        self.populate_model_combobox()
        self.modelComboBox.setMinimumWidth(175)
        gridLayout.addWidget(self.modelComboBox, 1, 0)

        self.sizeLabel = QLabel("—")
        self.sizeLabel.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.sizeLabel, 1, 1)

        self.vramLabel = QLabel("—")
        self.vramLabel.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.vramLabel, 1, 2)

        self.visionComponentLabel = QLabel("—")
        self.visionComponentLabel.setAlignment(Qt.AlignCenter)
        self.visionComponentLabel.setWordWrap(True)
        gridLayout.addWidget(self.visionComponentLabel, 1, 3)

        self.chatComponentLabel = QLabel("—")
        self.chatComponentLabel.setAlignment(Qt.AlignCenter)
        self.chatComponentLabel.setWordWrap(True)
        gridLayout.addWidget(self.chatComponentLabel, 1, 4)

        self.avgLenLabel = QLabel("—")
        self.avgLenLabel.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.avgLenLabel, 1, 5)

        cfg = _read_cfg()
        saved = fallback_if_unavailable((cfg.get("vision") or {}).get("chosen_model"), VISION_MODELS, DEFAULT_VISION_MODEL)
        self.modelComboBox.setCurrentText(saved)

        self.modelComboBox.currentTextChanged.connect(self._apply_model_to_labels)

        self._apply_model_to_labels(self.modelComboBox.currentText())

    def populate_model_combobox(self):
        self.modelComboBox.clear()
        self.modelComboBox.addItems([name for name, info in VISION_MODELS.items() if runs_on_this_hardware(info)])

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
        p = Path(CONFIG_FILE)
        if not cfg and p.exists() and p.stat().st_size > 0:
            return
        cfg.setdefault("vision", {})
        if cfg["vision"].get("chosen_model") != model_name:
            cfg["vision"]["chosen_model"] = model_name
            _write_cfg(cfg)
