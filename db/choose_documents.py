from pathlib import Path
from multiprocessing import Pool, cpu_count

import yaml
from PySide6.QtCore import QElapsedTimer, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFileSystemModel,
    QHBoxLayout,
    QProgressDialog,
    QVBoxLayout,
    QDialog,
    QTextEdit,
    QPushButton,
    QMessageBox,
)

from db.create_symlinks import _create_single_symlink
from core.constants import PROJECT_ROOT

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".epub",
    ".txt",
    ".enex",
    ".eml",
    ".msg",
    ".csv",
    ".xls",
    ".xlsx",
    ".rtf",
    ".odt",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".html",
    ".htm",
    ".md",
    ".doc",
}

DOCS_FOLDER = "Docs_for_DB"
CONFIG_FILE = "config.yaml"


class SymlinkWorker(QThread):
    progress = Signal(int)
    finished = Signal(int, list)

    def __init__(self, source, target_dir, parent=None):
        super().__init__(parent)
        self.source = source
        self.target_dir = Path(target_dir)

    def run(self):
        if isinstance(self.source, (str, Path)):
            dir_path = Path(self.source)
            try:
                import os
                filenames = os.listdir(str(dir_path))
                files = [
                    str(dir_path / filename)
                    for filename in filenames
                    if (dir_path / filename).is_file() 
                    and (dir_path / filename).suffix.lower() in ALLOWED_EXTENSIONS
                ]
            except OSError:
                files = []
                print(f"Error accessing directory {dir_path}")
        else:
            files = list(self.source)

        total = len(files)
        made = 0
        errors = []
        last_pct = -1
        timer = QElapsedTimer()
        timer.start()
        step = max(1, total // 100) if total else 1

        if total > 1000:
            processes = min((total // 10000) + 1, cpu_count())
            file_args = [(f, str(self.target_dir)) for f in files]
            with Pool(processes=processes) as pool:
                for i, (ok, err) in enumerate(
                    pool.imap_unordered(_create_single_symlink, file_args), 1
                ):
                    if ok:
                        made += 1
                    if err:
                        errors.append(err)
                    if i % step == 0 or i == total:
                        pct = int(i * 100 / total) if total else 100
                        if pct != last_pct and timer.elapsed() > 500:
                            self.progress.emit(pct)
                            last_pct = pct
                            timer.restart()
        else:
            for f in files:
                if self.isInterruptionRequested():
                    break

                ok, err = _create_single_symlink((f, str(self.target_dir)))
                if ok:
                    made += 1
                if err:
                    errors.append(err)
                if made % step == 0 or made == total:
                    pct = int(made * 100 / total) if total else 100
                    if pct != last_pct and timer.elapsed() > 500:
                        self.progress.emit(pct)
                        last_pct = pct
                        timer.restart()

        self.finished.emit(made, errors)


def choose_documents_directory():
    current_dir = PROJECT_ROOT
    target_dir = current_dir / DOCS_FOLDER
    target_dir.mkdir(parents=True, exist_ok=True)

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Selection Type")
    msg_box.setText("Would you like to select a directory or individual files?")

    dir_button = msg_box.addButton("Select Directory", QMessageBox.ActionRole)
    files_button = msg_box.addButton("Select Files", QMessageBox.ActionRole)
    cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

    msg_box.exec()
    clicked_button = msg_box.clickedButton()

    if clicked_button == cancel_button:
        return

    file_dialog = QFileDialog()

    def start_worker(source):
        progress = QProgressDialog(
            "Creating symlinks...", "Cancel", 0, 0
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        worker = SymlinkWorker(source, target_dir)
        main_window = _get_main_window()
        if main_window and hasattr(main_window, "databases_tab"):
            db_tab = main_window.databases_tab
            if hasattr(db_tab, "docs_model") and db_tab.docs_model:
                if hasattr(QFileSystemModel, "DontWatchForChanges"):
                    db_tab.docs_model.setOption(
                        QFileSystemModel.DontWatchForChanges, True
                    )
                if hasattr(db_tab, "docs_refresh"):
                    db_tab.docs_refresh.start()

        progress.canceled.connect(worker.requestInterruption)

        def update_progress(pct):
            if progress.maximum() == 0:
                progress.setRange(0, 100)
            progress.setValue(pct)

        worker.progress.connect(update_progress)

        def _done(count, errs):
            if main_window and hasattr(main_window, "databases_tab"):
                db_tab = main_window.databases_tab
                if hasattr(db_tab, "docs_refresh"):
                    db_tab.docs_refresh.stop()
                if hasattr(db_tab, "docs_model") and db_tab.docs_model:
                    if hasattr(db_tab.docs_model, "refresh"):
                        db_tab.docs_model.refresh()
                    elif hasattr(db_tab.docs_model, "reindex"):
                        db_tab.docs_model.reindex()
                    if hasattr(QFileSystemModel, "DontWatchForChanges"):
                        db_tab.docs_model.setOption(
                            QFileSystemModel.DontWatchForChanges, False
                        )

            progress.reset()
            msg = f"Created {count} symlinks"
            if errs:
                msg += f" – {len(errs)} errors (see console)"
                print(*errs, sep="\n")
            QMessageBox.information(None, "Symlinks", msg)

        worker.finished.connect(_done)
        worker.progress.connect(update_progress)
        worker.start()

        choose_documents_directory._symlink_thread = worker

    if clicked_button == dir_button:
        file_dialog.setFileMode(QFileDialog.Directory)
        file_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        selected_dir = file_dialog.getExistingDirectory(
            None, "Choose Directory for Database", str(current_dir)
        )
        if selected_dir:
            start_worker(Path(selected_dir))
    else:
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_paths = file_dialog.getOpenFileNames(
            None, "Choose Documents and Images for Database", str(current_dir)
        )[0]
        if file_paths:
            compatible_files = []
            incompatible_files = []
            for file_path in file_paths:
                path = Path(file_path)
                if path.suffix.lower() in ALLOWED_EXTENSIONS:
                    compatible_files.append(str(path))
                else:
                    incompatible_files.append(path.name)

            if incompatible_files and not show_incompatible_files_dialog(
                incompatible_files
            ):
                return

            if compatible_files:
                start_worker(compatible_files)


def show_incompatible_files_dialog(incompatible_files):
    dialog_text = (
        "The following files cannot be added here due to their file extension:\n\n"
        + "\n".join(incompatible_files)
        + "\n\nHowever, if any of them are audio files you can still add them directly in the Tools Tab."
        "\n\nClick 'Ok' to add the compatible documents only (remembering to add audio files separately)"
        " or 'Cancel' to back out completely."
    )

    incompatible_dialog = QDialog()
    incompatible_dialog.resize(800, 600)
    incompatible_dialog.setWindowTitle("Incompatible Files Detected")

    layout = QVBoxLayout()
    text_edit = QTextEdit()
    text_edit.setReadOnly(True)
    text_edit.setText(dialog_text)
    layout.addWidget(text_edit)

    button_box = QHBoxLayout()
    ok_button = QPushButton("OK")
    cancel_button = QPushButton("Cancel")
    button_box.addWidget(ok_button)
    button_box.addWidget(cancel_button)

    layout.addLayout(button_box)
    incompatible_dialog.setLayout(layout)

    ok_button.clicked.connect(incompatible_dialog.accept)
    cancel_button.clicked.connect(incompatible_dialog.reject)

    return incompatible_dialog.exec() == QDialog.Accepted


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def select_embedding_model_directory():
    initial_dir = Path("Models") if Path("Models").exists() else Path.home()
    chosen_directory = QFileDialog.getExistingDirectory(
        None, "Select Embedding Model Directory", str(initial_dir)
    )
    if chosen_directory:
        config_file_path = Path(CONFIG_FILE)
        config_data = (
            yaml.safe_load(config_file_path.read_text(encoding="utf-8"))
            if config_file_path.exists()
            else {}
        )
        config_data["EMBEDDING_MODEL_NAME"] = chosen_directory
        config_file_path.write_text(yaml.dump(config_data), encoding="utf-8")


def _get_main_window():
    for widget in QApplication.topLevelWidgets():
        if hasattr(widget, "databases_tab"):
            return widget
    return None
