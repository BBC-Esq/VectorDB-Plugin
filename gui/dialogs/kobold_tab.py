from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class KoboldTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        notice = QLabel(
            "Kobold settings will appear here in a future release.\n\n"
            "Kobold currently uses its default connection settings."
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
