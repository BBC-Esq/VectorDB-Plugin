from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class MiniMaxTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        notice = QLabel(
            "MiniMax settings will appear here in a future release.\n\n"
            "For now, the MiniMax API key is managed via the File menu's "
            "MiniMax API Key entry."
        )
        notice.setAlignment(Qt.AlignCenter)
        notice.setWordWrap(True)
        notice.setStyleSheet("color: #B0B0B0; font-style: italic; padding: 24px;")

        layout.addStretch(1)
        layout.addWidget(notice)
        layout.addStretch(2)

    def load_from_config(self, config: dict) -> None:
        return

    def save_to_config(self, config: dict) -> None:
        return

    def validate(self) -> tuple[bool, str | None]:
        return True, None
