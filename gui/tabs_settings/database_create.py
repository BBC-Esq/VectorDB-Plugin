import yaml
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QHBoxLayout, QComboBox, QCheckBox, QMessageBox

from core.constants import TOOLTIPS


class ChunkSettingsTab(QWidget):
    def __init__(self):
        super(ChunkSettingsTab, self).__init__()
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            self.database_config = config_data["database"]
            self.compute_device_options = config_data["Compute_Device"]["available"]
            self.database_creation_device = config_data["Compute_Device"]["database_creation"]

        preset_tooltip = (
            "Controls CPU parallelism during database creation.\n"
            "Minimal: sequential processing (1 thread/process)\n"
            "Low: light parallelism (2-4 workers)\n"
            "Normal: moderate parallelism (default)\n"
            "High: aggressive parallelism\n"
            "Maximum: all available CPU cores"
        )

        current_size = self.database_config.get("chunk_size", "")
        current_overlap = self.database_config.get("chunk_overlap", "")
        current_preset = self.database_config.get("pipeline_preset", "normal")

        self.device_label = QLabel("Device:")
        self.device_label.setToolTip(TOOLTIPS["CREATE_DEVICE_DB"])
        self.device_combo = QComboBox()
        self.device_combo.addItems(self.compute_device_options)
        self.device_combo.setToolTip(TOOLTIPS["CREATE_DEVICE_DB"])
        if self.database_creation_device in self.compute_device_options:
            self.device_combo.setCurrentIndex(
                self.compute_device_options.index(self.database_creation_device)
            )
        self.device_combo.setMinimumWidth(100)

        self.half_precision_label = QLabel("Half-Precision (2x speedup - GPU only):")
        self.half_precision_label.setToolTip(TOOLTIPS["HALF_PRECISION"])
        self.half_precision_checkbox = QCheckBox()
        self.half_precision_checkbox.setChecked(self.database_config.get("half", False))
        self.half_precision_checkbox.setToolTip(TOOLTIPS["HALF_PRECISION"])

        self.preset_label = QLabel("Pipeline Performance:")
        self.preset_label.setToolTip(preset_tooltip)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["minimal", "low", "normal", "high", "maximum"])
        self.preset_combo.setCurrentText(current_preset)
        self.preset_combo.setToolTip(preset_tooltip)
        self.preset_combo.setMinimumWidth(100)

        self.chunk_size_label = QLabel("Chunk Size (# characters):")
        self.chunk_size_label.setToolTip(TOOLTIPS["CHUNK_SIZE"])
        self.current_size_label = QLabel(f"{current_size}")
        self.current_size_label.setToolTip(TOOLTIPS["CHUNK_SIZE"])
        self.chunk_size_edit = QLineEdit()
        self.chunk_size_edit.setPlaceholderText("Enter new chunk_size...")
        self.chunk_size_edit.setValidator(QIntValidator(1, 1000000))
        self.chunk_size_edit.setToolTip(TOOLTIPS["CHUNK_SIZE"])

        self.chunk_overlap_label = QLabel("Chunk Overlap (# characters):")
        self.chunk_overlap_label.setToolTip(TOOLTIPS["CHUNK_OVERLAP"])
        self.current_overlap_label = QLabel(f"{current_overlap}")
        self.current_overlap_label.setToolTip(TOOLTIPS["CHUNK_OVERLAP"])
        self.chunk_overlap_edit = QLineEdit()
        self.chunk_overlap_edit.setPlaceholderText("Enter new chunk_overlap...")
        self.chunk_overlap_edit.setValidator(QIntValidator(0, 1000000))
        self.chunk_overlap_edit.setToolTip(TOOLTIPS["CHUNK_OVERLAP"])

        def labeled(label, current, editor, editor_stretch=1):
            box = QHBoxLayout()
            box.addWidget(label)
            box.addWidget(current)
            box.addWidget(editor, editor_stretch)
            return box

        device_cell = QHBoxLayout()
        device_cell.addWidget(self.device_label)
        device_cell.addWidget(self.device_combo, 1)

        half_cell = QHBoxLayout()
        half_cell.addWidget(self.half_precision_label)
        half_cell.addWidget(self.half_precision_checkbox)
        half_cell.addStretch(1)

        preset_cell = QHBoxLayout()
        preset_cell.addWidget(self.preset_label)
        preset_cell.addWidget(self.preset_combo, 1)

        size_cell = labeled(self.chunk_size_label, self.current_size_label, self.chunk_size_edit)
        overlap_cell = labeled(self.chunk_overlap_label, self.current_overlap_label, self.chunk_overlap_edit)

        grid_layout = QGridLayout()
        for col in range(6):
            grid_layout.setColumnStretch(col, 1)

        grid_layout.addLayout(device_cell, 0, 0, 1, 2)
        grid_layout.addLayout(preset_cell, 0, 2, 1, 2)
        grid_layout.addLayout(half_cell,   0, 4, 1, 2)
        grid_layout.addLayout(size_cell,    1, 0, 1, 3)
        grid_layout.addLayout(overlap_cell, 1, 3, 1, 3)

        self.setLayout(grid_layout)

    def update_config(self):
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"An error occurred while loading the configuration: {e}",
            )
            return False

        settings_changed = False
        errors = []

        new_device = self.device_combo.currentText()
        device_changed = new_device != self.database_creation_device

        new_chunk_size_text = self.chunk_size_edit.text().strip()
        if new_chunk_size_text:
            try:
                new_chunk_size = int(new_chunk_size_text)
                if new_chunk_size <= 0:
                    raise ValueError("Chunk size must be a positive integer.")
            except ValueError as ve:
                errors.append(f"Chunk size must be a positive integer: {str(ve)}")
        else:
            new_chunk_size = self.database_config.get("chunk_size", 0)

        new_chunk_overlap_text = self.chunk_overlap_edit.text().strip()
        if new_chunk_overlap_text:
            try:
                new_chunk_overlap = int(new_chunk_overlap_text)
                if new_chunk_overlap < 0:
                    raise ValueError("Chunk overlap cannot be negative.")
            except ValueError as ve:
                errors.append(
                    f"Chunk overlap must be a non-negative integer: {str(ve)}"
                )
        else:
            new_chunk_overlap = self.database_config.get("chunk_overlap", 0)

        if new_chunk_size and new_chunk_overlap >= new_chunk_size:
            errors.append("Chunk overlap must be less than chunk size.")

        if errors:
            error_message = "\n".join(errors)
            QMessageBox.warning(
                self, "Invalid Input", f"The following errors occurred:\n{error_message}"
            )
            return False

        if device_changed:
            config_data["Compute_Device"]["database_creation"] = new_device
            self.database_creation_device = new_device
            settings_changed = True

        if new_chunk_size_text and new_chunk_size != self.database_config.get(
            "chunk_size", 0
        ):
            config_data["database"]["chunk_size"] = new_chunk_size
            self.current_size_label.setText(f"{new_chunk_size}")
            settings_changed = True

        if new_chunk_overlap_text and new_chunk_overlap != self.database_config.get(
            "chunk_overlap", 0
        ):
            config_data["database"]["chunk_overlap"] = new_chunk_overlap
            self.current_overlap_label.setText(f"{new_chunk_overlap}")
            settings_changed = True

        new_half_precision = self.half_precision_checkbox.isChecked()
        if new_half_precision != self.database_config.get("half", False):
            config_data["database"]["half"] = new_half_precision
            settings_changed = True

        new_preset = self.preset_combo.currentText()
        if new_preset != self.database_config.get("pipeline_preset", "normal"):
            config_data["database"]["pipeline_preset"] = new_preset
            settings_changed = True

        if settings_changed:
            try:
                with open("config.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(config_data, f)

                self.database_config["chunk_size"] = config_data["database"]["chunk_size"]
                self.database_config["chunk_overlap"] = config_data["database"]["chunk_overlap"]
                self.database_config["half"] = config_data["database"]["half"]
                self.database_config["pipeline_preset"] = config_data["database"].get("pipeline_preset", "normal")

                self.database_creation_device = config_data["Compute_Device"][
                    "database_creation"
                ]

                self.chunk_overlap_edit.clear()
                self.chunk_size_edit.clear()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Configuration",
                    f"An error occurred while saving the configuration: {e}",
                )
                return False
        else:
            return False

        return settings_changed
