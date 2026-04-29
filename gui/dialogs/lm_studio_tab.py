import re

from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QCheckBox,
    QGroupBox,
)

from core.constants import TOOLTIPS

DEFAULT_CONNECTION_STR = "http://localhost:1234/v1"
PORT_RE = re.compile(r":(\d{1,5})(?=/)")


class LMStudioTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._original_connection_str = DEFAULT_CONNECTION_STR
        self._original_port = ""

        layout = QVBoxLayout(self)

        server_group = QGroupBox("LM Studio Server")
        server_layout = QVBoxLayout()

        port_row = QHBoxLayout()
        self.port_label = QLabel("Port:")
        self.port_label.setToolTip(TOOLTIPS.get("PORT", ""))
        self.port_edit = QLineEdit()
        self.port_edit.setPlaceholderText("Port")
        self.port_edit.setValidator(QIntValidator(1, 65535))
        self.port_edit.setToolTip(TOOLTIPS.get("PORT", ""))
        port_row.addWidget(self.port_label)
        port_row.addWidget(self.port_edit, 1)
        server_layout.addLayout(port_row)

        self.thinking_checkbox = QCheckBox("Show thinking process?")
        self.thinking_checkbox.setToolTip(TOOLTIPS.get("SHOW_THINKING_CHECKBOX", ""))
        server_layout.addWidget(self.thinking_checkbox)

        server_group.setLayout(server_layout)
        layout.addWidget(server_group)
        layout.addStretch(1)

    def load_from_config(self, config: dict) -> None:
        server_cfg = config.get("server") or {}
        self._original_connection_str = server_cfg.get("connection_str") or DEFAULT_CONNECTION_STR

        match = PORT_RE.search(self._original_connection_str)
        self._original_port = match.group(1) if match else ""

        self.port_label.setText(f"Port: {self._original_port}" if self._original_port else "Port:")
        self.port_edit.setText(self._original_port)

        self.thinking_checkbox.setChecked(bool(server_cfg.get("show_thinking", False)))

    def save_to_config(self, config: dict) -> None:
        server_cfg = config.setdefault("server", {})

        new_port_text = self.port_edit.text().strip()
        if new_port_text:
            new_connection_str = self._update_port_in_connection_str(
                self._original_connection_str, new_port_text
            )
            server_cfg["connection_str"] = new_connection_str
        else:
            server_cfg.setdefault("connection_str", self._original_connection_str)

        server_cfg["show_thinking"] = bool(self.thinking_checkbox.isChecked())

    def validate(self) -> tuple[bool, str | None]:
        new_port_text = self.port_edit.text().strip()
        if not new_port_text:
            return True, None
        try:
            port = int(new_port_text)
        except ValueError:
            return False, "Port must be a number between 1 and 65535."
        if not (1 <= port <= 65535):
            return False, "Port must be between 1 and 65535."

        if not PORT_RE.search(self._original_connection_str):
            return False, (
                "Existing LM Studio connection string is malformed and the port "
                "cannot be replaced. Edit config.yaml directly to fix it."
            )
        return True, None

    @staticmethod
    def _update_port_in_connection_str(connection_str: str, port: str) -> str:
        match = PORT_RE.search(connection_str)
        if not match:
            return connection_str
        return connection_str[: match.start(1)] + str(port) + connection_str[match.end(1):]
