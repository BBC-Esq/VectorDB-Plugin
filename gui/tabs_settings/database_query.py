import yaml
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QLineEdit,
    QGridLayout,
    QSizePolicy,
    QComboBox,
    QPushButton,
    QMessageBox,
)

from core.constants import TOOLTIPS


class DatabaseSettingsTab(QWidget):
    def __init__(self):
        super(DatabaseSettingsTab, self).__init__()

        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                self.database_config = config_data["database"]
                self.compute_device_options = config_data["Compute_Device"]["available"]
                self.database_query_device = config_data["Compute_Device"]["database_query"]
                self.search_term = self.database_config.get("search_term", "")
                self.document_type = self.database_config.get("document_types", "")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"An error occurred while loading the configuration: {e}",
            )
            self.database_config = {}
            self.compute_device_options = []
            self.database_query_device = ""
            self.search_term = ""
            self.document_type = ""

        grid_layout = QGridLayout()

        self.field_data = {}
        self.label_data = {}

        self.query_device_label = QLabel(f"Device: {self.database_query_device}")
        self.query_device_combo = QComboBox()
        self.query_device_combo.addItems(self.compute_device_options)
        self.query_device_combo.setToolTip(TOOLTIPS["CREATE_DEVICE_QUERY"])
        if self.database_query_device in self.compute_device_options:
            self.query_device_combo.setCurrentIndex(
                self.compute_device_options.index(self.database_query_device)
            )
        grid_layout.addWidget(self.query_device_label, 0, 0)
        grid_layout.addWidget(self.query_device_combo, 0, 1)

        similarity_value = self.database_config.get("similarity", "")
        self.similarity_edit = QLineEdit()
        self.similarity_edit.setPlaceholderText("Similarity (0.0 - 1.0)...")
        self.similarity_edit.setValidator(QDoubleValidator(0.0, 1.0, 4))
        self.similarity_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.similarity_edit.setToolTip(TOOLTIPS["SIMILARITY"])
        self.similarity_label = QLabel(f"Similarity: {similarity_value}")
        self.similarity_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.similarity_label.setToolTip(TOOLTIPS["SIMILARITY"])
        grid_layout.addWidget(self.similarity_label, 1, 0)
        grid_layout.addWidget(self.similarity_edit, 1, 1)
        self.field_data["similarity"] = self.similarity_edit
        self.label_data["similarity"] = self.similarity_label

        contexts_value = self.database_config.get("contexts", "")
        self.contexts_edit = QLineEdit()
        self.contexts_edit.setPlaceholderText("# Contexts to return...")
        self.contexts_edit.setValidator(QIntValidator(1, 1000000))
        self.contexts_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.contexts_edit.setToolTip(TOOLTIPS["CONTEXTS"])
        self.contexts_label = QLabel(f"Contexts: {contexts_value}")
        self.contexts_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.contexts_label.setToolTip(TOOLTIPS["CONTEXTS"])
        grid_layout.addWidget(self.contexts_label, 1, 3)
        grid_layout.addWidget(self.contexts_edit, 1, 4)
        self.field_data["contexts"] = self.contexts_edit
        self.label_data["contexts"] = self.contexts_label

        self.search_term_edit = QLineEdit()
        self.search_term_edit.setPlaceholderText("Term to exclude...")
        self.search_term_edit.setText(self.search_term)
        self.search_term_edit.setToolTip(TOOLTIPS["SEARCH_TERM_FILTER"])
        self.search_term_label = QLabel(f"Search Term Filter: {self.search_term}")
        self.search_term_label.setToolTip(TOOLTIPS["SEARCH_TERM_FILTER"])
        self.filter_button = QPushButton("Clear Filter")
        self.filter_button.clicked.connect(self.reset_search_term)
        grid_layout.addWidget(self.search_term_label, 2, 0)
        grid_layout.addWidget(self.search_term_edit, 2, 1)
        grid_layout.addWidget(self.filter_button, 2, 2)

        self.file_type_combo = QComboBox()
        file_type_items = ["All Files", "Images Only", "Documents Only", "Audio Only"]
        self.file_type_combo.addItems(file_type_items)
        self.file_type_combo.setToolTip(TOOLTIPS["FILE_TYPE_FILTER"])

        if self.document_type == "image":
            default_index = file_type_items.index("Images Only")
        elif self.document_type == "document":
            default_index = file_type_items.index("Documents Only")
        elif self.document_type == "audio":
            default_index = file_type_items.index("Audio Only")
        else:
            default_index = file_type_items.index("All Files")
        self.file_type_combo.setCurrentIndex(default_index)

        self.file_type_label = QLabel("File Type:")
        self.file_type_label.setToolTip(TOOLTIPS["FILE_TYPE_FILTER"])
        grid_layout.addWidget(self.file_type_label, 2, 3)
        grid_layout.addWidget(self.file_type_combo, 2, 4)

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

        new_query_device = self.query_device_combo.currentText()
        device_changed = new_query_device != config_data["Compute_Device"].get(
            "database_query", ""
        )

        new_similarity_text = self.similarity_edit.text().strip()
        if new_similarity_text:
            try:
                new_similarity = float(new_similarity_text)
                if not (0.0 <= new_similarity <= 1.0):
                    raise ValueError("Similarity must be between 0.0 and 1.0.")
            except ValueError:
                errors.append("Similarity must be a number between 0.0 and 1.0.")
        else:
            new_similarity = self.database_config.get("similarity", 0.0)

        new_contexts_text = self.contexts_edit.text().strip()
        if new_contexts_text:
            try:
                new_contexts = int(new_contexts_text)
                if new_contexts < 1:
                    raise ValueError("Contexts must be a positive integer.")
            except ValueError:
                errors.append("Contexts must be a positive integer.")
        else:
            new_contexts = self.database_config.get("contexts", 1)

        new_search_term = self.search_term_edit.text().strip()

        file_type_map = {
            "All Files": "",
            "Images Only": "image",
            "Documents Only": "document",
            "Audio Only": "audio",
        }

        file_type_selection = self.file_type_combo.currentText()
        document_type_value = file_type_map.get(file_type_selection, "")

        if errors:
            error_message = "\n".join(errors)
            QMessageBox.warning(
                self, "Invalid Input", f"The following errors occurred:\n{error_message}"
            )
            return False

        if device_changed:
            config_data["Compute_Device"]["database_query"] = new_query_device
            settings_changed = True

        if new_similarity_text and new_similarity != config_data["database"].get(
            "similarity", 0.0
        ):
            config_data["database"]["similarity"] = new_similarity
            settings_changed = True

        if new_contexts_text and new_contexts != config_data["database"].get(
            "contexts", 1
        ):
            config_data["database"]["contexts"] = new_contexts
            settings_changed = True

        if new_search_term and new_search_term != config_data["database"].get(
            "search_term", ""
        ):
            config_data["database"]["search_term"] = new_search_term
            settings_changed = True

        if document_type_value != config_data["database"].get("document_types", ""):
            config_data["database"]["document_types"] = document_type_value
            settings_changed = True

        if settings_changed:
            try:
                with open("config.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(config_data, f)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Configuration",
                    f"An error occurred while saving the configuration: {e}",
                )
                return False

            if device_changed:
                self.database_query_device = new_query_device
                self.query_device_label.setText(f"Device: {new_query_device}")

            if new_similarity_text:
                self.database_config["similarity"] = new_similarity
                self.similarity_label.setText(f"Similarity: {new_similarity}")

            if new_contexts_text:
                self.database_config["contexts"] = new_contexts
                self.contexts_label.setText(f"Contexts: {new_contexts}")

            if new_search_term:
                self.search_term = new_search_term
                self.database_config["search_term"] = new_search_term
                self.search_term_label.setText(f"Search Term Filter: {new_search_term}")

            self.document_type = document_type_value
            self.database_config["document_types"] = document_type_value
            self.file_type_label.setText("File Type: " + file_type_selection)

            self.similarity_edit.clear()
            self.contexts_edit.clear()
            self.search_term_edit.clear()

        return settings_changed

    def reset_search_term(self):
        try:
            with open("config.yaml", "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"An error occurred while loading the configuration: {e}",
            )
            return

        config_data["database"]["search_term"] = ""

        try:
            with open("config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Configuration",
                f"An error occurred while saving the configuration: {e}",
            )
            return

        self.search_term = ""
        self.database_config["search_term"] = ""
        self.search_term_label.setText("Search Term Filter: ")
        self.search_term_edit.clear()
