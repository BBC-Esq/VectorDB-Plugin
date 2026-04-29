from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QFrame,
)

from core.chatgpt_settings import (
    AVAILABLE_OPENAI_MODELS,
    REASONING_EFFORT_OPTIONS,
    VERBOSITY_OPTIONS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_VERBOSITY,
    DEFAULT_REASONING_EFFORT,
    get_display_name,
    get_model_pricing,
    supports_reasoning_effort,
    supports_verbosity,
    migrate_legacy_model,
)


class CostPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setStyleSheet("""
            CostPanel {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
            }
            QLabel { color: #E8E8E8; }
            QLabel#costHeader { font-weight: bold; color: #2196F3; }
            QLabel#costValue { font-family: monospace; color: #4CAF50; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        header = QLabel("API Cost (per million tokens)")
        header.setObjectName("costHeader")
        layout.addWidget(header)

        cost_row = QHBoxLayout()
        cost_row.setSpacing(20)

        self.input_value = self._build_cost_column(cost_row, "Input:")
        self.cached_value = self._build_cost_column(cost_row, "Cached:")
        self.output_value = self._build_cost_column(cost_row, "Output:")
        cost_row.addStretch()

        layout.addLayout(cost_row)

    def _build_cost_column(self, parent_layout: QHBoxLayout, label_text: str) -> QLabel:
        column = QVBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 9pt; color: #B0B0B0;")
        value = QLabel("$0.00")
        value.setObjectName("costValue")
        value.setStyleSheet("font-size: 11pt; font-weight: bold;")
        column.addWidget(label)
        column.addWidget(value)
        parent_layout.addLayout(column)
        return value

    def update_for_model(self, model_name: str) -> None:
        input_cost, cached_cost, output_cost = get_model_pricing(model_name)
        paid_style = "font-size: 11pt; font-weight: bold; color: #FFA726;"
        muted_style = "font-size: 11pt; font-weight: bold; color: #B0B0B0;"

        self.input_value.setText(f"${input_cost:.2f}")
        self.input_value.setStyleSheet(paid_style)

        self.output_value.setText(f"${output_cost:.2f}")
        self.output_value.setStyleSheet(paid_style)

        if cached_cost > 0:
            self.cached_value.setText(f"${cached_cost:.3f}")
            self.cached_value.setStyleSheet(paid_style)
        else:
            self.cached_value.setText("—")
            self.cached_value.setStyleSheet(muted_style)


class ChatGPTTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        api_group = QGroupBox("API Key")
        api_layout = QVBoxLayout()
        api_help = QLabel(
            "<small>Required for ChatGPT. Get a key from "
            "<a href='https://platform.openai.com/api-keys'>platform.openai.com/api-keys</a>.</small>"
        )
        api_help.setOpenExternalLinks(True)
        api_help.setStyleSheet("color: gray;")
        api_layout.addWidget(api_help)

        api_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-proj-...")
        self.show_key_btn = QPushButton("Show / Hide")
        self.show_key_btn.setMaximumWidth(110)
        self.show_key_btn.clicked.connect(self._toggle_api_key_visibility)
        api_row.addWidget(self.api_key_edit)
        api_row.addWidget(self.show_key_btn)
        api_layout.addLayout(api_row)
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)

        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for model in AVAILABLE_OPENAI_MODELS:
            self.model_combo.addItem(get_display_name(model), model)
        model_row.addWidget(self.model_combo, 1)
        model_layout.addLayout(model_row)

        self.cost_panel = CostPanel()
        model_layout.addWidget(self.cost_panel)

        verbosity_row = QHBoxLayout()
        self.verbosity_label = QLabel("Verbosity:")
        self.verbosity_combo = QComboBox()
        self.verbosity_combo.addItems(VERBOSITY_OPTIONS)
        verbosity_row.addWidget(self.verbosity_label)
        verbosity_row.addWidget(self.verbosity_combo, 1)
        model_layout.addLayout(verbosity_row)

        reasoning_row = QHBoxLayout()
        self.reasoning_label = QLabel("Reasoning Effort:")
        self.reasoning_combo = QComboBox()
        self.reasoning_combo.addItems(REASONING_EFFORT_OPTIONS)
        reasoning_row.addWidget(self.reasoning_label)
        reasoning_row.addWidget(self.reasoning_combo, 1)
        model_layout.addLayout(reasoning_row)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        layout.addStretch(1)

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)

    def _toggle_api_key_visibility(self) -> None:
        if self.api_key_edit.echoMode() == QLineEdit.Password:
            self.api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_edit.setEchoMode(QLineEdit.Password)

    def _on_model_changed(self) -> None:
        model = self.model_combo.currentData() or self.model_combo.currentText()
        self.cost_panel.update_for_model(model)
        self._update_capability_visibility(model)

    def _update_capability_visibility(self, model: str) -> None:
        show_v = supports_verbosity(model)
        show_r = supports_reasoning_effort(model)
        self.verbosity_label.setVisible(show_v)
        self.verbosity_combo.setVisible(show_v)
        self.reasoning_label.setVisible(show_r)
        self.reasoning_combo.setVisible(show_r)

    def _set_combo_to_model(self, model: str) -> None:
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == model:
                self.model_combo.setCurrentIndex(i)
                return
        self.model_combo.setCurrentIndex(0)

    def load_from_config(self, config: dict) -> None:
        openai_cfg = (config.get("openai") or {})
        api_key = openai_cfg.get("api_key") or ""
        self.api_key_edit.setText(api_key)

        model = migrate_legacy_model(openai_cfg.get("model") or DEFAULT_OPENAI_MODEL)
        self._set_combo_to_model(model)

        verbosity = openai_cfg.get("verbosity") or DEFAULT_VERBOSITY
        if verbosity in VERBOSITY_OPTIONS:
            self.verbosity_combo.setCurrentText(verbosity)
        else:
            self.verbosity_combo.setCurrentText(DEFAULT_VERBOSITY)

        reasoning = openai_cfg.get("reasoning_effort") or DEFAULT_REASONING_EFFORT
        if reasoning in REASONING_EFFORT_OPTIONS:
            self.reasoning_combo.setCurrentText(reasoning)
        else:
            self.reasoning_combo.setCurrentText(DEFAULT_REASONING_EFFORT)

        self.cost_panel.update_for_model(model)
        self._update_capability_visibility(model)

    def save_to_config(self, config: dict) -> None:
        openai_cfg = config.setdefault("openai", {})
        openai_cfg["api_key"] = self.api_key_edit.text().strip() or None
        openai_cfg["model"] = self.model_combo.currentData() or self.model_combo.currentText()
        openai_cfg["verbosity"] = self.verbosity_combo.currentText()
        openai_cfg["reasoning_effort"] = self.reasoning_combo.currentText()

    def validate(self) -> tuple[bool, str | None]:
        return True, None
