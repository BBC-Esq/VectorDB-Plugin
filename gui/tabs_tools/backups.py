import sqlite3
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from core.initialize import restore_vector_db_backup
from core.utilities import backup_database
from db.sqlite_operations import stamps_match

JEEVES_DATABASE = "user_manual"


class WorkerThread(QThread):
    finished = Signal(bool, str)

    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.function(*self.args, **self.kwargs)
            self.finished.emit(True, "")
        except Exception as e:
            print(f"Error during {self.function.__name__}: {e}")
            self.finished.emit(False, str(e))


def database_names(folder):
    try:
        return {p.name for p in Path(folder).iterdir() if p.is_dir() and p.name != JEEVES_DATABASE}
    except OSError:
        return set()


def backup_has_content():
    try:
        return any(Path('Vector_DB_Backup').iterdir())
    except OSError:
        return False


def other_database_work():
    if QApplication.instance() is None:
        return None
    for widget in QApplication.allWidgets():
        task_running = getattr(widget, "database_task_running", None)
        if callable(task_running) and task_running():
            return "A database task is running on the Manage Databases tab. Wait for it to finish."
        in_progress = getattr(widget, "database_in_progress", None)
        if callable(in_progress) and in_progress():
            return "A database is being created on the Create Database tab. Wait for it to finish."
    return None


def out_of_date(names):
    stale = []
    for name in names:
        try:
            if not stamps_match(Path('Vector_DB') / name, Path('Vector_DB_Backup') / name, timeout=1):
                stale.append(name)
        except (OSError, sqlite3.Error):
            continue
    return sorted(stale, key=str.lower)


class BackupTool(QObject):
    changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.worker = None
        self.task = None
        self.started = None
        self.result = None
        self.databases = set()
        self.backed_up = set()
        self.stale = []
        self.has_backup = False
        self.refresh()

    def refresh(self):
        self.databases = database_names('Vector_DB')
        self.backed_up = database_names('Vector_DB_Backup')
        self.stale = out_of_date(self.databases & self.backed_up)
        self.has_backup = backup_has_content()

    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def blocked(self):
        return None if self.running() else other_database_work()

    def state(self):
        return {
            "databases": len(self.databases),
            "missing": sorted(self.databases - self.backed_up, key=str.lower),
            "stale": self.stale,
            "has_backup": self.has_backup,
            "blocked": self.blocked(),
            "task": self.task if self.running() else None,
            "started": self.started,
            "result": self.result,
        }

    def make_worker(self, function):
        return WorkerThread(function)

    def _start(self, task, function):
        self.task = task
        self.result = None
        self.started = time.time()
        self.worker = self.make_worker(function)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.changed.emit()

    def _must_wait(self, parent_widget):
        reason = other_database_work()
        if reason:
            QMessageBox.information(parent_widget, "Please Wait", reason)
        return reason

    def backup(self, parent_widget):
        if self.running() or self._must_wait(parent_widget):
            return
        confirm = QMessageBox.question(
            parent_widget,
            "Confirm Backup",
            "Warning. This will erase any existing backups and overwrite them with the current state of the \"Vector_DB\" folder.\n\nAre you sure you want to proceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self._start("backup", backup_database)

    def restore(self, parent_widget):
        if self.running() or self._must_wait(parent_widget):
            return
        confirm = QMessageBox.question(
            parent_widget,
            "Confirm Restoration",
            "Warning. This will overwrite current databases with the backup. Are you sure you want to proceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self._start("restore", restore_vector_db_backup)

    def _on_finished(self, success, message):
        worker = self.worker
        if worker is not None:
            worker.wait()
            self.worker = None
        task = self.task
        self.refresh()
        if task == "backup":
            if success:
                self.result = {"ok": True, "message": "All databases have been successfully backed up."}
            else:
                self.result = {"ok": False, "message": f"Failed to backup the databases. {message}"}
        else:
            if success:
                self.result = {"ok": True, "message": "The databases have been successfully restored from the backup."}
            else:
                self.result = {"ok": False, "message": f"The databases were not restored and are unchanged. {message}"}
        self.changed.emit()

    def busy_message(self):
        if self.running():
            return f"A database {self.task} is still running. Please wait for it to finish before closing the program."
        return None
