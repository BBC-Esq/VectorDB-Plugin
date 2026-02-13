import yaml
from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QGroupBox, QVBoxLayout, QSizePolicy


class ModelsSettingsTab(QWidget):
    def __init__(self):
        super(ModelsSettingsTab, self).__init__()

        # Use explicit UTF-8 for consistency
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        main_layout = QVBoxLayout()
        self.field_data = {}
        self.label_data = {}

        # Only expose categories you want editable in the UI
        for category, sub_dict in config_data["embedding-models"].items():
            if category in ["bge", "instructor"]:
                group_box = QGroupBox(category)
                layout = QGridLayout()

                row = 0
                for setting, current_value in sub_dict.items():
                    full_key = f"{category}-{setting}"

                    edit = QLineEdit()
                    edit.setPlaceholderText(f"Enter new {setting.lower()}...")
                    edit.textChanged.connect(self.validate_model_token)
                    edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

                    label = QLabel(f"{setting}: {current_value}")
                    label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

                    layout.addWidget(edit, row, 0)
                    layout.addWidget(label, row + 1, 0)

                    self.field_data[full_key] = edit
                    self.label_data[full_key] = label
                    row += 2

                group_box.setLayout(layout)
                group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                main_layout.addWidget(group_box)

        self.setLayout(main_layout)

    def validate_model_token(self, text: str):
        """
        Allow common model-name characters while still preventing accidental spaces or punctuation:
        letters, digits, dash, underscore.
        """
        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        )
        if any(ch not in allowed for ch in text):
            sender = self.sender()
            # Strip disallowed characters but keep cursor behavior simple
            filtered = "".join(ch for ch in text if ch in allowed)
            sender.setText(filtered)

    def update_config(self) -> bool:
        # Read with explicit UTF-8
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        settings_changed = False

        for full_key, widget in self.field_data.items():
            # Only split once in case 'setting' contains hyphens (e.g., model names)
            category, setting = full_key.split("-", 1)
            new_value = widget.text().strip()

            if new_value and new_value != config_data["embedding-models"][category][setting]:
                settings_changed = True
                config_data["embedding-models"][category][setting] = new_value
                # Reflect in the UI immediately
                self.label_data[full_key].setText(f"{setting}: {new_value}")
                widget.clear()

        if settings_changed:
            with open("config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f)

        return settings_changed
