import gc
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from core.constants import PROJECT_ROOT
from core.utilities import _needs_ocr_worker, backup_database, check_preconditions_for_db_creation, my_cprint, save_config_atomically
from db.database_interactions import BUILD_COMPLETE_MARKER, DB_FOLDER_CREATED_MARKER, NOT_ADDED_MARKER

LOG_LINES = 400
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

STAGES = [
    ("extract", "Reading"),
    ("media", "Images and audio"),
    ("split", "Splitting"),
    ("embed", "Embedding"),
    ("write", "Saving"),
]
STAGE_INDEX = {key: i for i, (key, _) in enumerate(STAGES)}


class VectorDBWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool, int, str)

    def __init__(self, database_name, parent=None):
        super().__init__(parent)
        self.database_name = database_name
        self._process = None
        self._cancelled = False
        self.not_added = []
        self.created_folder = False
        self.completed = False

    def run(self):
        temp_root = None
        try:
            temp_root = tempfile.mkdtemp(prefix="vectordb_build_")
            cmd = [
                sys.executable, "-c",
                "from db.database_interactions import create_vector_db_in_process; "
                f"create_vector_db_in_process({self.database_name!r})"
            ]

            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "TMPDIR": temp_root}

            self.progress.emit("Initializing database creation...")

            if self._cancelled:
                self.finished.emit(False, -1, "Cancelled by user.")
                return

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
            if self._cancelled:
                self._terminate_process_tree(self._process.pid)

            for line in self._process.stdout:
                line = line.rstrip("\n")
                if line.startswith(NOT_ADDED_MARKER):
                    try:
                        self.not_added = json.loads(line[len(NOT_ADDED_MARKER):])
                    except ValueError:
                        pass
                    continue
                if line == DB_FOLDER_CREATED_MARKER:
                    self.created_folder = True
                    continue
                if line == BUILD_COMPLETE_MARKER:
                    self.completed = True
                    continue
                if line.strip():
                    try:
                        print(f"  [DB Creation] {line}", flush=True)
                    except UnicodeEncodeError:
                        print(f"  [DB Creation] {line}".encode("ascii", "replace").decode("ascii"), flush=True)
                    self.progress.emit(line)

            self._process.wait()
            exit_code = self._process.returncode

            if exit_code == 0 or self.completed:
                result = (True, exit_code, "The database finished before the cancel took effect, so it was kept."
                          if self._cancelled else "Database created successfully!")
            elif self._cancelled:
                result = (False, exit_code, "Cancelled by user.")
            else:
                result = (
                    False, exit_code,
                    f"Database build failed (exit code {exit_code}). "
                    "Check the log window for details."
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            result = (False, -1, f"Database creation failed: {e}")
        finally:
            proc = self._process
            if proc and proc.poll() is None:
                try:
                    self._terminate_process_tree(proc.pid)
                except Exception:
                    pass
            if temp_root:
                self._remove_temp_root(temp_root)

        self.finished.emit(*result)

    @staticmethod
    def _remove_temp_root(path):
        for _ in range(8):
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                return
            time.sleep(0.25)

    def cancel(self):
        self._cancelled = True
        proc = self._process
        if proc and proc.poll() is None:
            self._terminate_process_tree(proc.pid)

    @staticmethod
    def _terminate_process_tree(pid):
        import psutil
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        try:
            procs = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            procs = []
        procs.append(parent)
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(procs, timeout=5)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass


def ocr_threads():
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or mp.cpu_count()
    except ImportError:
        logical = mp.cpu_count()
        physical = max(1, max(logical // 2, logical - 4))
    return max(1, physical - 1)


class OcrScanWorker(QThread):
    progress = Signal(int, int)
    scanned = Signal(list)

    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.folder = Path(folder)

    def run(self):
        try:
            with os.scandir(self.folder) as it:
                pdf_paths = [Path(entry.path) for entry in it if entry.name.lower().endswith(".pdf")]
        except OSError:
            pdf_paths = []
        total = len(pdf_paths)
        flags = [False] * total
        done = 0
        self.progress.emit(0, total)
        with ThreadPoolExecutor(max_workers=ocr_threads()) as ex:
            futures = {ex.submit(_needs_ocr_worker, str(path)): i for i, path in enumerate(pdf_paths)}
            for future in as_completed(futures):
                if self.isInterruptionRequested():
                    ex.shutdown(wait=False, cancel_futures=True)
                    self.scanned.emit([])
                    return
                flags[futures[future]] = bool(future.result())
                done += 1
                self.progress.emit(done, total)
        self.scanned.emit([str(path) for path, flag in zip(pdf_paths, flags) if flag])


class BackupWorker(QThread):
    done = Signal(bool, str)

    def __init__(self, database_name, parent=None):
        super().__init__(parent)
        self.database_name = database_name

    def run(self):
        try:
            backup_database(self.database_name)
            self.done.emit(True, "")
        except Exception as e:
            self.done.emit(False, str(e))


class BuildProgress:
    PATTERNS = [
        (re.compile(r"Extracting documents"), "stage", "extract"),
        (re.compile(r"Extracted ([\d,]+) documents"), "documents", None),
        (re.compile(r"Processing any (audio transcripts|images)"), "stage", "media"),
        (re.compile(r"Splitting documents into chunks"), "stage", "split"),
        (re.compile(r"Split into ([\d,]+) chunks"), "chunks", None),
        (re.compile(r"Computing vectors"), "stage", "embed"),
        (re.compile(r"Running forward pass on (\d+) pre-padded batches"), "batches", None),
        (re.compile(r"Forward pass: (\d+)/(\d+) batches"), "batch", None),
        (re.compile(r"Forward pass complete"), "embedded", None),
        (re.compile(r"Creating TileDB array|Writing TileDB array|Write attempt"), "stage", "write"),
        (re.compile(r"Database created\. Total time"), "stage", "done"),
    ]

    def __init__(self):
        self.stage = "start"
        self.documents = None
        self.chunks = None
        self.batches = None
        self.batch = None
        self.log = []

    def feed(self, line):
        line = ANSI.sub("", line).replace("\r", " ").strip()
        if not line:
            return False
        self.log.append(line)
        if len(self.log) > LOG_LINES:
            del self.log[: len(self.log) - LOG_LINES]
        for pattern, kind, value in self.PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            if kind == "stage":
                if value == "done" or STAGE_INDEX.get(value, -1) >= STAGE_INDEX.get(self.stage, -1):
                    self.stage = value
            elif kind == "documents":
                self.documents = int(match.group(1).replace(",", ""))
            elif kind == "chunks":
                self.chunks = int(match.group(1).replace(",", ""))
            elif kind == "batches":
                self.batches = int(match.group(1))
                self.batch = 0
            elif kind == "batch":
                self.batch, self.batches = int(match.group(1)), int(match.group(2))
            elif kind == "embedded" and self.batches:
                self.batch = self.batches
            return True
        return False

    def state(self):
        current = STAGE_INDEX.get(self.stage, -1)
        if self.stage == "done":
            current = len(STAGES)
        return {
            "stage": self.stage,
            "stages": [
                {"key": key, "label": label,
                 "status": "done" if i < current else "running" if i == current else "pending"}
                for i, (key, label) in enumerate(STAGES)
            ],
            "documents": self.documents,
            "chunks": self.chunks,
            "batches": self.batches,
            "batch": self.batch,
            "line": self.log[-1] if self.log else "",
        }


def transcription_running():
    return any(callable(getattr(w, "transcription_running", None)) and w.transcription_running()
               for w in QApplication.allWidgets())


def update_config_with_database_name(database_name):
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file) or {}
        model = config.get('EMBEDDING_MODEL_NAME')
        chunk_size = config.get('database', {}).get('chunk_size')
        chunk_overlap = config.get('database', {}).get('chunk_overlap')
        if 'created_databases' not in config or not isinstance(config['created_databases'], dict):
            config['created_databases'] = {}
        config['created_databases'][database_name] = {
            'model': model,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
        save_config_atomically(config, config_path, allow_unicode=True)


def confirm_cpu_creation(parent_widget, cfg):
    compute = cfg.get('Compute_Device') or {}
    available = compute.get('available') or []
    if compute.get('database_creation') != 'cpu' or not any(d in available for d in ('cuda', 'mps')):
        return True
    reply = QMessageBox.question(
        parent_widget,
        "GPU Acceleration Available",
        "GPU acceleration is available and strongly recommended for creating a vector database, "
        "but the database creation device is set to CPU.\n\n"
        "Create this database on the CPU anyway?\n\n"
        "Choose No to change the device in the Settings tab first.",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No
    )
    return reply == QMessageBox.Yes


class BuildController(QObject):
    changed = Signal()

    def __init__(self, parent, files):
        super().__init__(parent)
        self.files = files
        self.db_worker = None
        self.scan_worker = None
        self.backup_worker = None
        self.phase = "idle"
        self.name = None
        self.model_name = None
        self.model_display = None
        self.started = None
        self.progress = None
        self.scan = None
        self.result = None
        self.cancelling = False

    def running(self):
        return self.db_worker is not None and self.db_worker.isRunning()

    def busy(self):
        return self.phase in ("scanning", "building", "saving")

    def state(self):
        return {
            "phase": self.phase,
            "name": self.name,
            "model": self.model_display or self.model_name,
            "started": self.started,
            "progress": self.progress.state() if self.progress else None,
            "scan": self.scan,
            "cancelling": self.cancelling,
            "result": self.result,
        }

    def log(self):
        return self.progress.log if self.progress else []

    def _fail(self, message, kind="error"):
        self.phase = "idle"
        self.result = {"kind": kind, "message": message}
        self.changed.emit()
        return {"error": message}

    def start(self, parent_widget, database_name, model_label, check_pdfs, cfg, model_display=None):
        if self.busy():
            return {"error": "A database is already being created."}
        if self.files.staging_running():
            return self._fail("Files are still being added to the list. Create the database after that finishes "
                              "so every file is included.")
        if transcription_running():
            return self._fail("An audio transcription is still running. Create the database after it finishes "
                              "so the transcript is included.")
        if not cfg.get("EMBEDDING_MODEL_NAME"):
            return self._fail("Please select a model before creating a database.")
        database_name = (database_name or "").strip()
        if not database_name:
            return self._fail("Please enter a database name before creating a database.")
        docs = PROJECT_ROOT / "Docs_for_DB"
        if not docs.exists() or not any(p for p in docs.iterdir() if p.is_file()):
            return self._fail("The Docs_for_DB folder is empty. Add at least one file before creating a database.")
        ok, msg = check_preconditions_for_db_creation(PROJECT_ROOT, database_name, skip_ocr=True)
        if not ok:
            return self._fail(msg)
        if not confirm_cpu_creation(parent_widget, cfg):
            return {"cancelled": True}
        self.name = database_name
        self.model_name = model_label
        self.model_display = model_display or model_label
        self.result = None
        self.cancelling = False
        has_pdfs = any(p.suffix.lower() == ".pdf" for p in docs.iterdir() if p.is_file())
        if has_pdfs and check_pdfs:
            self._start_scan(docs)
        else:
            self._start_build()
        return {"ok": True}

    def make_scan_worker(self, folder):
        return OcrScanWorker(folder)

    def _start_scan(self, docs):
        self.phase = "scanning"
        self.started = time.time()
        self.scan = {"done": 0, "total": 0}
        self.progress = None
        self.scan_worker = self.make_scan_worker(docs)
        self.scan_worker.progress.connect(self._on_scan_progress)
        self.scan_worker.scanned.connect(self._on_scanned)
        self.scan_worker.start()
        self.changed.emit()

    def _on_scan_progress(self, done, total):
        if self.phase == "scanning":
            self.scan = {"done": done, "total": total}
            self.changed.emit()

    def _on_scanned(self, needs_ocr):
        worker = self.scan_worker
        if worker is not None:
            worker.wait()
            self.scan_worker = None
        if self.cancelling:
            self.phase = "idle"
            self.cancelling = False
            self.result = {"kind": "cancelled", "message": "The PDF check was cancelled, so no database was created."}
            self.changed.emit()
            return
        if needs_ocr:
            self.phase = "idle"
            self.result = {"kind": "ocr", "paths": needs_ocr,
                           "names": [Path(p).name for p in needs_ocr]}
            self.changed.emit()
            return
        self._start_build()

    def make_worker(self, database_name):
        return VectorDBWorker(database_name, parent=self)

    def _start_build(self):
        self.phase = "building"
        self.started = time.time()
        self.progress = BuildProgress()
        self.db_worker = self.make_worker(self.name)
        self.db_worker.progress.connect(self._on_line)
        self.db_worker.finished.connect(self._on_finished)
        self.db_worker.start()
        my_cprint(f"Started database creation for: {self.name}", "green")
        self.changed.emit()

    def _on_line(self, line):
        if self.progress is not None:
            self.progress.feed(line)
            self.changed.emit()

    def cancel(self):
        if self.phase == "scanning" and self.scan_worker is not None:
            self.cancelling = True
            self.scan_worker.requestInterruption()
            self.changed.emit()
        elif self.running():
            self.cancelling = True
            self.db_worker.cancel()
            self.changed.emit()

    def _on_finished(self, success, exit_code, message):
        was_cancelled = (not success) and message == "Cancelled by user."
        worker = self.db_worker
        elapsed = time.time() - (self.started or time.time())
        try:
            if was_cancelled:
                if self.name:
                    partial_dir = PROJECT_ROOT / "Vector_DB" / self.name
                    if partial_dir.exists():
                        shutil.rmtree(partial_dir, ignore_errors=True)
                self.result = {"kind": "cancelled",
                               "message": "Database creation was cancelled and any partial files were removed."}
            elif success:
                my_cprint(f"{self.model_name} removed from memory.", "red")
                update_config_with_database_name(self.name)
                not_added = worker.not_added if worker is not None else []
                self.result = {
                    "kind": "ok",
                    "name": self.name,
                    "message": message,
                    "seconds": round(elapsed, 1),
                    "documents": self.progress.documents if self.progress else None,
                    "chunks": self.progress.chunks if self.progress else None,
                    "not_added": not_added,
                    "backup": "running",
                }
                self._start_backup(self.name)
            else:
                if self.name and worker is not None and worker.created_folder:
                    partial_dir = PROJECT_ROOT / "Vector_DB" / self.name
                    if partial_dir.exists():
                        shutil.rmtree(partial_dir, ignore_errors=True)
                self.result = {"kind": "error", "message": message, "log": True}
        except Exception as e:
            self.result = {"kind": "error", "message": f"Error handling completion: {e}"}
        finally:
            if worker is not None:
                worker.wait()
                worker.deleteLater()
                self.db_worker = None
            self.cancelling = False
            if self.phase != "saving":
                self.phase = "idle"
            self.files.refresh(force=True)
            gc.collect()
            self.changed.emit()

    def make_backup_worker(self, database_name):
        return BackupWorker(database_name)

    def _start_backup(self, database_name):
        self.phase = "saving"
        self.backup_worker = self.make_backup_worker(database_name)
        self.backup_worker.done.connect(self._on_backup_done)
        self.backup_worker.start()

    def _on_backup_done(self, ok, message):
        worker = self.backup_worker
        if worker is not None:
            worker.wait()
            self.backup_worker = None
        if self.result and self.result.get("kind") == "ok":
            self.result = {**self.result, "backup": "ok" if ok else "failed", "backup_error": message}
        self.phase = "idle"
        self.changed.emit()

    def dismiss(self):
        if not self.busy():
            self.result = None
            self.progress = None if self.phase == "idle" else self.progress
            self.changed.emit()

    def open_report(self):
        paths = (self.result or {}).get("paths") or []
        if not paths:
            return
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write("PDFs that need OCR:\n\n")
            for pdf_path in paths:
                tmp.write(f"{pdf_path}\n")
            temp_path = tmp.name
        os.startfile(temp_path)

        def cleanup():
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
        threading.Timer(1.0, cleanup).start()

    def remove_ocr_files(self):
        names = (self.result or {}).get("names") or []
        removed = self.files.remove(names)
        self.result = {"kind": "info", "message": f"Removed {removed} PDF file(s) that need OCR from the list."}
        self.changed.emit()

    def busy_message(self):
        if self.phase == "saving":
            return "The new database is still being backed up. Please wait for it to finish before closing the program."
        return None

    def cleanup(self):
        if self.scan_worker is not None and self.scan_worker.isRunning():
            self.scan_worker.requestInterruption()
            self.scan_worker.wait(5000)
        if self.db_worker is not None and self.db_worker.isRunning():
            self.db_worker.cancel()
            self.db_worker.wait(5000)
            if self.name:
                if self.db_worker.completed:
                    update_config_with_database_name(self.name)
                else:
                    partial_dir = PROJECT_ROOT / "Vector_DB" / self.name
                    if partial_dir.exists():
                        shutil.rmtree(partial_dir, ignore_errors=True)
        if self.backup_worker is not None and self.backup_worker.isRunning():
            self.backup_worker.wait()
