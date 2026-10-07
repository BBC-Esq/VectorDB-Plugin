import os
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox

from core.constants import PROJECT_ROOT
from core.utilities import open_file
from db.choose_documents import ALLOWED_EXTENSIONS, SymlinkWorker

KINDS = [
    ("pdf", "PDF", {".pdf"}),
    ("word", "Word", {".docx"}),
    ("text", "Text", {".txt", ".md", ".rtf"}),
    ("web", "Web pages", {".html", ".htm"}),
    ("email", "Email", {".eml", ".msg"}),
    ("sheet", "Spreadsheets", {".csv", ".xls", ".xlsx", ".xlsm"}),
    ("image", "Images", {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}),
    ("transcript", "Transcripts", {".json"}),
]
KIND_BY_SUFFIX = {suffix: kind for kind, _, suffixes in KINDS for suffix in suffixes}
TARGET_LIMIT = 3000


def docs_dir():
    return PROJECT_ROOT / "Docs_for_DB"


def kind_of(name):
    return KIND_BY_SUFFIX.get(Path(name).suffix.lower(), "other")


def scan_docs(folder):
    try:
        with os.scandir(folder) as it:
            return sorted(entry.name for entry in it if not entry.is_dir(follow_symlinks=False))
    except OSError:
        return []


class StagedFiles(QObject):
    changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.names = []
        self.version = 0
        self._mtime = None
        self.worker = None
        self.staging = None
        self.notice = None
        self.refresh(force=True)

    def refresh(self, force=False):
        folder = docs_dir()
        try:
            mtime = folder.stat().st_mtime if folder.exists() else None
        except OSError:
            mtime = None
        if not force and mtime == self._mtime:
            return
        self._mtime = mtime
        names = scan_docs(folder) if mtime is not None else []
        if names != self.names:
            self.names = names
            self.version += 1
            self.changed.emit()

    def summary(self):
        counts = {}
        for name in self.names:
            kind = kind_of(name)
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    def entries(self):
        folder = docs_dir()
        with_targets = len(self.names) <= TARGET_LIMIT
        rows = []
        for name in self.names:
            row = {"name": name, "kind": kind_of(name)}
            if with_targets:
                path = folder / name
                try:
                    target = os.readlink(path) if path.is_symlink() else str(path)
                    row["target"] = target[4:] if target.startswith("\\\\?\\") else target
                except OSError:
                    pass
            rows.append(row)
        return rows

    def state(self):
        summary = self.summary()
        return {
            "version": self.version,
            "count": len(self.names),
            "summary": summary,
            "staging": self.staging,
            "notice": self.notice,
        }

    def staging_running(self):
        return self.worker is not None and self.worker.isRunning()

    def add_files(self, parent_widget):
        if self.staging_running():
            return
        file_paths = QFileDialog.getOpenFileNames(
            parent_widget, "Choose Documents and Images for Database", str(PROJECT_ROOT)
        )[0]
        if not file_paths:
            return
        compatible = [str(Path(p)) for p in file_paths if Path(p).suffix.lower() in ALLOWED_EXTENSIONS]
        skipped = [Path(p).name for p in file_paths if Path(p).suffix.lower() not in ALLOWED_EXTENSIONS]
        self.notice = None
        if skipped:
            self.notice = {"kind": "warn", "skipped": skipped,
                           "message": f"{len(skipped)} file(s) can't be added because of their file type."}
        if compatible:
            self.start_staging(compatible, skipped)
        else:
            self.changed.emit()

    def add_folder(self, parent_widget):
        if self.staging_running():
            return
        selected_dir = QFileDialog.getExistingDirectory(parent_widget, "Choose Directory for Database", str(PROJECT_ROOT))
        if not selected_dir:
            return
        selected_path = Path(selected_dir)
        top_level_files = [
            str(p) for p in selected_path.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        subdirectory_files = [
            str(p) for p in selected_path.rglob("*")
            if p.is_file()
            and p.parent != selected_path
            and p.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        include_subdirs = False
        if subdirectory_files:
            reply = QMessageBox.question(
                parent_widget,
                "Include Subdirectories?",
                (
                    f"This folder contains {len(top_level_files)} compatible file(s) "
                    f"at the top level and {len(subdirectory_files)} more in "
                    f"subdirectories.\n\nInclude the subdirectory files as well?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            include_subdirs = reply == QMessageBox.Yes
        files = top_level_files + subdirectory_files if include_subdirs else top_level_files
        if files:
            self.notice = None
            self.start_staging(files, [])
        else:
            self.notice = {"kind": "info", "message": "No compatible files were found in the selected folder."}
            self.changed.emit()

    def make_worker(self, files, target_dir):
        return SymlinkWorker(files, target_dir)

    def start_staging(self, files, skipped):
        target = docs_dir()
        target.mkdir(parents=True, exist_ok=True)
        self.staging = {"total": len(files), "percent": 0, "cancelling": False}
        self._skipped = skipped
        self.worker = self.make_worker(files, target)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self.changed.emit()

    def _on_progress(self, percent):
        if self.staging is not None:
            self.staging = {**self.staging, "percent": int(percent)}
            self.changed.emit()

    def _on_finished(self, count, errors):
        worker = self.worker
        if worker is not None:
            worker.wait()
            self.worker = None
        total = self.staging["total"] if self.staging else count
        cancelled = bool(self.staging and self.staging.get("cancelling"))
        self.staging = None
        if errors:
            print(*errors, sep="\n")
        already = max(0, total - count - len(errors)) if not cancelled else 0
        parts = [f"Added {count:,} file{'s' if count != 1 else ''} to the list."]
        if already:
            parts.append(f"{already:,} {'was' if already == 1 else 'were'} already in it.")
        if cancelled:
            parts.append("Stopped before the rest were added.")
        kind = "warn" if errors or self._skipped else "ok"
        self.notice = {"kind": kind, "message": " ".join(parts), "errors": errors[:200], "error_count": len(errors),
                       "skipped": self._skipped}
        self.refresh(force=True)
        self.changed.emit()

    def cancel_staging(self):
        if self.staging_running():
            self.staging = {**self.staging, "cancelling": True}
            self.worker.requestInterruption()
            self.changed.emit()

    def remove(self, names):
        folder = docs_dir()
        failed = []
        removed = 0
        for name in names:
            if not name or "/" in name or "\\" in name:
                continue
            path = folder / name
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
            except OSError as e:
                failed.append(f"{name}: {e.strerror or e}")
        if failed:
            self.notice = {"kind": "error", "message": f"{len(failed)} file(s) could not be removed. Remove them manually.",
                           "errors": failed}
        self.refresh(force=True)
        self.changed.emit()
        return removed

    def open(self, name):
        if name and "/" not in name and "\\" not in name:
            open_file(str(docs_dir() / name))

    def dismiss_notice(self):
        self.notice = None
        self.changed.emit()

    def cleanup(self):
        if self.staging_running():
            self.worker.requestInterruption()
            self.worker.wait(5000)
