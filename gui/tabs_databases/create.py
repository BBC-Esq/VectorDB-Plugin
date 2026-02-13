import time
import gc
import json
from pathlib import Path
import multiprocessing
import yaml
from PySide6.QtCore import QDir, QRegularExpression, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QRegularExpressionValidator
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QTreeView, QFileSystemModel, QMenu, QGroupBox, QLineEdit, QGridLayout, QSizePolicy, QComboBox

from db.database_interactions import create_vector_db_in_process
from db.choose_documents import choose_documents_directory
from core.utilities import check_preconditions_for_db_creation, open_file, delete_file, backup_database, my_cprint
from gui.download_model import model_downloaded_signal
from core.constants import TOOLTIPS, PROJECT_ROOT


class CreateDatabaseProcess:
    def __init__(self, database_name, parent=None):
        self.database_name = database_name
        self.process = None

    def start(self):
        self.process = multiprocessing.Process(target=create_vector_db_in_process, args=(self.database_name,))
        self.process.start()

    def wait(self, timeout=None):
        if self.process:
            self.process.join(timeout)

    def is_alive(self):
        if self.process:
            return self.process.is_alive()
        return False

    def terminate(self):
        if self.process and self.process.is_alive():
            self.process.terminate()


class CustomFileSystemModel(QFileSystemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilter(QDir.Files)


class DatabasesTab(QWidget):
    def __init__(self):
        super().__init__()
        model_downloaded_signal.downloaded.connect(self.update_model_combobox)
        self.layout = QVBoxLayout(self)
        self.documents_group_box = self.create_group_box("Files To Add to Database", "Docs_for_DB")
        self.groups = {self.documents_group_box: 1}
        grid_layout_top_buttons = QGridLayout()
        self.choose_docs_button = QPushButton("Choose Files")
        self.choose_docs_button.setToolTip(TOOLTIPS["CHOOSE_FILES"])
        self.choose_docs_button.clicked.connect(choose_documents_directory)
        self.model_combobox = QComboBox()
        self.model_combobox.setToolTip(TOOLTIPS["SELECT_VECTOR_MODEL"])
        self.populate_model_combobox()
        self.model_combobox.currentIndexChanged.connect(self.on_model_selected)
        self.model_combobox.activated.connect(self.refresh_model_combobox)
        self.create_db_button = QPushButton("Create Vector Database")
        self.create_db_button.setToolTip(TOOLTIPS["CREATE_VECTOR_DB"])
        self.create_db_button.clicked.connect(self.on_create_db_clicked)
        self.create_db_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grid_layout_top_buttons.addWidget(self.choose_docs_button, 0, 0)
        grid_layout_top_buttons.addWidget(self.model_combobox, 0, 1)
        grid_layout_top_buttons.addWidget(self.create_db_button, 0, 2)
        number_of_columns = 3
        for column_index in range(number_of_columns):
            grid_layout_top_buttons.setColumnStretch(column_index, 1)
        hbox2 = QHBoxLayout()
        self.database_name_input = QLineEdit()
        self.database_name_input.setToolTip(TOOLTIPS["DATABASE_NAME_INPUT"])
        self.database_name_input.setPlaceholderText("Enter database name")
        self.database_name_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        regex = QRegularExpression("^[a-z0-9_-]*$")
        validator = QRegularExpressionValidator(regex, self.database_name_input)
        self.database_name_input.setValidator(validator)
        hbox2.addWidget(self.database_name_input)
        self.layout.addLayout(grid_layout_top_buttons)
        self.layout.addLayout(hbox2)
        self.sync_combobox_with_config()
        
        self.db_process = None
        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.check_process_status)
        self.current_model_name = None
        self.current_database_name = None

    def _validation_failed(self, message: str):
        QMessageBox.warning(self, "Validation Failed", message)
        self.reenable_create_db_button()

    def refresh_model_combobox(self, index):
        current_text = self.model_combobox.currentText()
        self.populate_model_combobox()
        idx = self.model_combobox.findText(current_text)
        if idx >= 0:
            self.model_combobox.setCurrentIndex(idx)

    def update_model_combobox(self, model_name, model_type):
        if model_type == "vector":
            self.populate_model_combobox()
            self.sync_combobox_with_config()

    def populate_model_combobox(self):
        self.model_combobox.clear()
        self.model_combobox.addItem("Select a model", None)
        script_dir = PROJECT_ROOT
        vector_dir = script_dir / "Models" / "vector"
        if not vector_dir.exists():
            return
        for folder in vector_dir.iterdir():
            if folder.is_dir():
                display_name = folder.name
                full_path = str(folder)
                self.model_combobox.addItem(display_name, full_path)

    def sync_combobox_with_config(self):
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file) or {}
            current_model = config_data.get("EMBEDDING_MODEL_NAME")
            if current_model:
                model_index = self.model_combobox.findData(current_model)
                if model_index != -1:
                    self.model_combobox.setCurrentIndex(model_index)
                else:
                    self.model_combobox.setCurrentIndex(0)
            else:
                self.model_combobox.setCurrentIndex(0)
        else:
            self.model_combobox.setCurrentIndex(0)

    def on_model_selected(self, index):
        selected_path = self.model_combobox.itemData(index)
        config_path = PROJECT_ROOT / "config.yaml"
        config_data = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file) or {}
        if selected_path:
            config_data["EMBEDDING_MODEL_NAME"] = selected_path
            if "stella" in selected_path.lower() or "static-retrieval" in selected_path.lower():
                config_data["EMBEDDING_MODEL_DIMENSIONS"] = 1024
            else:
                config_json_path = Path(selected_path) / "config.json"
                if config_json_path.exists():
                    with open(config_json_path, 'r', encoding='utf-8') as json_file:
                        model_config = json.load(json_file)
                    embedding_dimensions = model_config.get("hidden_size") or model_config.get("d_model")
                    if embedding_dimensions and isinstance(embedding_dimensions, int):
                        config_data["EMBEDDING_MODEL_DIMENSIONS"] = embedding_dimensions
        else:
            config_data.pop("EMBEDDING_MODEL_NAME", None)
            config_data.pop("EMBEDDING_MODEL_DIMENSIONS", None)
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(config_data, file, allow_unicode=True)

    def create_group_box(self, title, directory_name):
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        tree_view = self.setup_directory_view(directory_name)
        layout.addWidget(tree_view)
        group_box.setLayout(layout)
        self.layout.addWidget(group_box)
        group_box.toggled.connect(lambda checked, gb=group_box: self.toggle_group_box(gb, checked))
        return group_box

    def _refresh_docs_model(self):
        if hasattr(self.docs_model, 'refresh'):
            self.docs_model.refresh()
        elif hasattr(self.docs_model, 'reindex'):
            self.docs_model.reindex()

    def setup_directory_view(self, directory_name):
        tree_view = QTreeView()
        model = CustomFileSystemModel()
        tree_view.setModel(model)
        tree_view.setSelectionMode(QTreeView.ExtendedSelection)
        script_dir = PROJECT_ROOT
        directory_path = script_dir / directory_name
        model.setRootPath(str(directory_path))
        tree_view.setRootIndex(model.index(str(directory_path)))
        tree_view.hideColumn(1)
        tree_view.hideColumn(2)
        tree_view.hideColumn(3)
        tree_view.doubleClicked.connect(self.on_double_click)
        tree_view.setContextMenuPolicy(Qt.CustomContextMenu)
        tree_view.customContextMenuRequested.connect(self.on_context_menu)
        if directory_name == "Docs_for_DB":
            self.docs_model = model
            self.docs_view = tree_view
            self.docs_refresh = QTimer(self)
            self.docs_refresh.setInterval(500)
            self.docs_refresh.timeout.connect(self._refresh_docs_model)
        return tree_view

    def on_double_click(self, index):
        tree_view = self.sender()
        model = tree_view.model()
        file_path = model.filePath(index)
        open_file(file_path)

    def on_context_menu(self, point):
        tree_view = self.sender()
        context_menu = QMenu(self)
        delete_action = QAction("Delete File", self)
        context_menu.addAction(delete_action)
        delete_action.triggered.connect(lambda: self.on_delete_file(tree_view))
        context_menu.exec_(tree_view.viewport().mapToGlobal(point))

    def on_delete_file(self, tree_view):
        selected_indexes = tree_view.selectedIndexes()
        model = tree_view.model()
        for index in selected_indexes:
            if index.column() == 0:
                file_path = model.filePath(index)
                delete_file(file_path)

    def on_create_db_clicked(self):
        if self.model_combobox.currentIndex() == 0:
            QMessageBox.warning(self, "No Model Selected", "Please select a model before creating a database.")
            return

        self.create_db_button.setDisabled(True)
        self.choose_docs_button.setDisabled(True)
        self.model_combobox.setDisabled(True)
        self.database_name_input.setDisabled(True)

        database_name = self.database_name_input.text().strip()
        model_name = self.model_combobox.currentText()

        self.current_database_name = database_name
        self.current_model_name = model_name

        docs_dir = PROJECT_ROOT / "Docs_for_DB"
        has_pdfs = any(p.suffix.lower() == ".pdf" for p in docs_dir.iterdir() if p.is_file())
        skip_ocr = False
        if has_pdfs:
            reply = QMessageBox.question(self, "OCR Check",
                                         "PDF files detected. Do you want to check if any of the PDFs need OCR? "
                                         "If there are a lot of PDFs, it is time-consuming but strongly recommended.",
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)
            skip_ocr = (reply == QMessageBox.No)

        self.start_database_creation(database_name, model_name, skip_ocr)

    def start_database_creation(self, database_name, model_name, skip_ocr):
        try:
            script_dir = PROJECT_ROOT
            ok, msg = check_preconditions_for_db_creation(script_dir, database_name, skip_ocr=skip_ocr)
            if not ok:
                self._validation_failed(msg)
                return

            self.db_process = CreateDatabaseProcess(database_name)
            self.db_process.start()

            self.process_timer.start(500)
            
            my_cprint(f"Started database creation for: {database_name}", "green")

        except Exception as e:
            self._validation_failed(f"Failed to start database creation: {str(e)}")

    def check_process_status(self):
        if not self.db_process:
            self.process_timer.stop()
            return

        if self.db_process.is_alive():
            return

        self.process_timer.stop()

        try:
            self.db_process.wait(timeout=5.0)
            
            exit_code = self.db_process.process.exitcode

            if hasattr(self.db_process.process, 'close'):
                self.db_process.process.close()

            if exit_code == 0:
                my_cprint(f"{self.current_model_name} removed from memory.", "red")
                self.update_config_with_database_name()
                backup_database(self.current_database_name)
                QMessageBox.information(self, "Success", "Database created successfully!")

            else:
                err_msg = (f"Database build failed (exit code {exit_code}). "
                          "Check the log window for details.")
                QMessageBox.critical(self, "Error", err_msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error checking process status: {str(e)}")

        finally:
            self.db_process = None
            self.reenable_create_db_button()

    def update_config_with_database_name(self):
        config_path = PROJECT_ROOT / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file) or {}
            model = config.get('EMBEDDING_MODEL_NAME')
            chunk_size = config.get('database', {}).get('chunk_size')
            chunk_overlap = config.get('database', {}).get('chunk_overlap')
            if 'created_databases' not in config or not isinstance(config['created_databases'], dict):
                config['created_databases'] = {}
            config['created_databases'][self.current_database_name] = {
                'model': model,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
            with open(config_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(config, file, allow_unicode=True)

    def reenable_create_db_button(self):
        self.create_db_button.setDisabled(False)
        self.choose_docs_button.setDisabled(False)
        self.model_combobox.setDisabled(False)
        self.database_name_input.setDisabled(False)
        
        self.current_database_name = None
        self.current_model_name = None
        
        gc.collect()

    def closeEvent(self, event):
        if self.db_process and self.db_process.is_alive():
            self.db_process.terminate()
            self.db_process.wait()
        if hasattr(self, 'process_timer'):
            self.process_timer.stop()
        event.accept()

    def toggle_group_box(self, group_box, checked):
        self.groups[group_box] = 1 if checked else 0
        self.adjust_stretch()

    def adjust_stretch(self):
        for group, stretch in self.groups.items():
            self.layout.setStretchFactor(group, stretch if group.isChecked() else 0)
