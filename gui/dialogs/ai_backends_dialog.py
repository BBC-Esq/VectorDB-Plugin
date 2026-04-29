from pathlib import Path

import yaml
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QMessageBox,
)

from core.constants import PROJECT_ROOT
from gui.dialogs.chatgpt_tab import ChatGPTTab
from gui.dialogs.lm_studio_tab import LMStudioTab
from gui.dialogs.minimax_tab import MiniMaxTab
from gui.dialogs.kobold_tab import KoboldTab


class AIBackendsDialog(QDialog):
    TAB_REGISTRY = [
        ("ChatGPT", ChatGPTTab),
        ("LM Studio", LMStudioTab),
        ("MiniMax", MiniMaxTab),
        ("Kobold", KoboldTab),
    ]

    def __init__(self, parent=None, initial_tab=0):
        super().__init__(parent)
        self.setWindowTitle("Chat Backend Settings")
        self.resize(620, 540)

        self.config_path = PROJECT_ROOT / "config.yaml"
        config = self._load_config()

        self.tab_widget = QTabWidget()
        self.tabs = []

        for label, tab_class in self.TAB_REGISTRY:
            tab = tab_class()
            tab.load_from_config(config)
            self.tab_widget.addTab(tab, label)
            self.tabs.append(tab)

        if 0 <= initial_tab < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(initial_tab)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(ok_btn)
        button_row.addWidget(cancel_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tab_widget)
        layout.addLayout(button_row)

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config.yaml: {e}")
            return {}

    def _save_config(self, config: dict) -> bool:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config.yaml: {e}")
            return False

    def _on_accept(self) -> None:
        for idx, tab in enumerate(self.tabs):
            ok, error = tab.validate()
            if not ok:
                self.tab_widget.setCurrentIndex(idx)
                QMessageBox.warning(self, "Invalid Setting", error or "Validation failed.")
                return

        config = self._load_config()
        for tab in self.tabs:
            tab.save_to_config(config)

        if self._save_config(config):
            self.accept()
