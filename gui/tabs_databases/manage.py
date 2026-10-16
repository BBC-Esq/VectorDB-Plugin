import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QFileDialog, QTabWidget

from core.constants import PROJECT_ROOT
from core.utilities import open_file
from db.database_additions import ALLOWED_EXTENSIONS
from gui.web_common.web_tab import WebTab
from gui.tabs_databases import manage_data

WEB_PAGE = Path(__file__).resolve().parent / "web" / "manage_tab.html"
TASK_MESSAGES = {
    "delete": "A database is still being deleted. Please wait for it to finish before closing the program.",
    "backup": "A database is still being backed up. Please wait for it to finish before closing the program.",
    "restore": "A database is still being restored. Please wait for it to finish before closing the program.",
    "rebuild": "A database index is still being rebuilt. Please wait for it to finish before closing the program.",
    "remove_document": "A document is still being removed. Please wait for it to finish before closing the program.",
    "add_documents": "Documents are still being added. Please wait for the task to finish before closing the program.",
}


class InfoWorker(QThread):
    loaded = Signal(str, object, object)

    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.names = list(names)

    def run(self):
        for name in self.names:
            if self.isInterruptionRequested():
                return
            sig = manage_data.signature(name)
            try:
                info = manage_data.database_info(name, self.isInterruptionRequested)
            except Exception as e:
                info = {"files": None, "chunks": None, "dimensions": None, "size": None, "created": None,
                        "complete": False, "problem": str(e)}
            self.loaded.emit(name, sig, info)


class FilesWorker(QThread):
    loaded = Signal(int, object, str)
    checked = Signal(int, object)

    def __init__(self, token, name, parent=None):
        super().__init__(parent)
        self.token = token
        self.name = name

    def run(self):
        try:
            rows = manage_data.file_rows(self.name, self.isInterruptionRequested)
        except Exception as e:
            if not self.isInterruptionRequested():
                self.loaded.emit(self.token, None, str(e))
            return
        compact = manage_data.compact_rows(rows)
        summary = {}
        for row in compact["rows"]:
            summary[row[1]] = summary.get(row[1], 0) + 1
        self.loaded.emit(self.token, {"rows": rows, "payload": json.dumps(compact), "summary": summary}, "")
        missing = manage_data.missing_rows(rows, self.isInterruptionRequested)
        if missing is not None:
            self.checked.emit(self.token, missing)


class TaskWorker(QThread):
    done = Signal(str, str, object, str)

    def __init__(self, kind, name, readers=(), parent=None):
        super().__init__(parent)
        self.kind = kind
        self.name = name
        self.readers = list(readers)

    def run(self):
        for reader in self.readers:
            reader.wait(15000)
        try:
            if self.kind == "delete":
                result = manage_data.delete_database(self.name)
            elif self.kind == "backup":
                manage_data.backup_database(self.name)
                result = {}
            elif self.kind == "rebuild":
                result = manage_data.rebuild_index(self.name)
            else:
                manage_data.restore_database(self.name)
                result = {}
        except Exception as e:
            self.done.emit(self.kind, self.name, None, str(e))
            return
        self.done.emit(self.kind, self.name, result, "")


class DocumentMutationWorker(QThread):
    done = Signal(str, str, object, str)

    def __init__(self, database_name, document_name, document_id, parent=None):
        super().__init__(parent)
        self.database_name = database_name
        self.document_name = document_name
        self.document_id = document_id

    def run(self):
        script = PROJECT_ROOT / "db" / "stage_mutate.py"
        database_path = manage_data.vector_root() / self.database_name
        command = [sys.executable, str(script), "remove", str(database_path), str(self.document_id)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=3600,
            )
        except Exception as error:
            self.done.emit(self.database_name, self.document_name, None, f"{type(error).__name__}: {error}")
            return

        output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        marker = "VECTORDB_MUTATION_RESULT "
        result = None
        for line in completed.stdout.splitlines():
            if line.startswith(marker):
                try:
                    result = json.loads(line[len(marker):])
                except (TypeError, ValueError):
                    pass
        if result is not None:
            self.done.emit(self.database_name, self.document_name, result, "")
            return
        errors = [line.removeprefix("ERROR: ") for line in output.splitlines() if line.strip()]
        self.done.emit(
            self.database_name,
            self.document_name,
            None,
            errors[-1] if errors else "Document removal failed.",
        )


class DocumentAdditionWorker(QThread):
    done = Signal(str, object, str)
    progress = Signal(str)

    def __init__(self, database_name, paths, parent=None):
        super().__init__(parent)
        self.database_name = database_name
        self.paths = list(paths)

    def run(self):
        script = PROJECT_ROOT / "db" / "stage_mutate.py"
        database_path = manage_data.vector_root() / self.database_name
        try:
            with tempfile.TemporaryDirectory(prefix="vectordb_add_request_") as temporary:
                manifest = Path(temporary) / "manifest.json"
                manifest.write_text(json.dumps({"database_name": self.database_name, "paths": self.paths}), encoding="utf-8")
                process = subprocess.Popen(
                    [sys.executable, str(script), "add", str(database_path), str(manifest)],
                    cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
                )
                from db.subprocess_utils import drain_subprocess
                def on_line(line):
                    if line.startswith("VECTORDB_ADD_STAGE "):
                        self.progress.emit(line.removeprefix("VECTORDB_ADD_STAGE "))
                lines = drain_subprocess(process, 7200, on_line=on_line)
        except Exception as error:
            self.done.emit(self.database_name, None, f"{type(error).__name__}: {error}")
            return
        marker = "VECTORDB_MUTATION_RESULT "
        for line in lines:
            if line.startswith(marker):
                try:
                    self.done.emit(self.database_name, json.loads(line[len(marker):]), "")
                    return
                except ValueError:
                    pass
        errors = [line.removeprefix("ERROR: ") for line in lines if line.strip()]
        self.done.emit(self.database_name, None, errors[-1] if errors else "Document addition failed.")


def external_state():
    building = None
    tools_busy = False
    for widget in QApplication.allWidgets():
        in_progress = getattr(widget, "database_in_progress", None)
        if building is None and callable(in_progress):
            building = in_progress()
        backup_running = getattr(widget, "database_backup_running", None)
        if callable(backup_running) and backup_running():
            tools_busy = True
    return {"building": building, "tools_busy": tools_busy}


class ManageDatabasesTab(WebTab):
    def __init__(self, parent=None):
        super().__init__(WEB_PAGE, parent)
        self.entries = []
        self.config_error = None
        self.info = {}
        self.selected = None
        self.files = None
        self._files_token = 0
        self._info_worker = None
        self._pending_info = []
        self._workers = set()
        self.task = None
        self.notice = None
        self._next_selection = None
        self._pending_add = None
        self._skipped_add = []
        self.external = {"building": None, "tools_busy": False}
        self._stamp = None
        self._active = False
        self._poll = QTimer(self)
        self._poll.setInterval(1000)
        self._poll.timeout.connect(self._tick)
        self.refresh()

    def showEvent(self, event):
        self._active = True
        self.external = external_state()
        self.refresh()
        self._poll.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._poll.stop()
        super().hideEvent(event)

    def _stamps(self):
        stamps = []
        for path in (manage_data.config_path(), manage_data.vector_root(), manage_data.backup_root()):
            try:
                st = path.stat()
                stamps.append((st.st_mtime_ns, st.st_size))
            except OSError:
                stamps.append(None)
        return tuple(stamps)

    def _tick(self):
        external = external_state()
        moved = external != self.external
        self.external = external
        if moved or self._stamps() != self._stamp:
            self.refresh()

    def _start(self, worker):
        self._workers.add(worker)
        worker.finished.connect(lambda w=worker: self._workers.discard(w))
        worker.start()

    def refresh(self):
        self._stamp = self._stamps()
        try:
            cfg = manage_data.read_config()
            self.config_error = None
        except manage_data.ConfigError as e:
            cfg = {}
            self.config_error = str(e)
        self.entries = manage_data.list_databases(cfg)
        names = set()
        stale = []
        for entry in self.entries:
            entry["model"] = manage_data.model_summary(entry["model_path"])
            name = entry["name"]
            names.add(name)
            if not entry["folder"] or name == self.external["building"] or name in (self._replacing(), self._mutating()):
                continue
            cached = self.info.get(name)
            if cached is None or cached[0] != manage_data.signature(name):
                stale.append(name)
        self.info = {name: value for name, value in self.info.items() if name in names}
        if self.selected not in names:
            self.selected = self.entries[0]["name"] if self.entries else None
        self._queue_info(stale)
        self._ensure_files()
        self.schedule_push()

    def _queue_info(self, names):
        if not self._active:
            return
        for name in names:
            if name not in self._pending_info:
                self._pending_info.append(name)
        if self._info_worker is None and self._pending_info:
            batch, self._pending_info = self._pending_info, []
            self._info_worker = InfoWorker(batch)
            self._info_worker.loaded.connect(self._on_info)
            self._info_worker.finished.connect(self._on_info_finished)
            self._start(self._info_worker)

    def _on_info(self, name, sig, info):
        if any(entry["name"] == name for entry in self.entries):
            self.info[name] = (sig, info)
            self.schedule_push()

    def _on_info_finished(self):
        interrupted = self._info_worker is not None and self._info_worker.isInterruptionRequested()
        self._info_worker = None
        if interrupted:
            self.refresh()
        else:
            self._queue_info([])

    def _replacing(self):
        return self.task["name"] if self.task and self.task["kind"] in ("delete", "rebuild") else None

    def _mutating(self):
        return self.task["name"] if self.task and self.task["kind"] in ("remove_document", "add_documents") else None

    def _entry(self, name):
        return next((entry for entry in self.entries if entry["name"] == name), None)

    def _ensure_files(self):
        if not self._active:
            return
        name = self.selected
        entry = self._entry(name)
        if (not entry or not entry["folder"] or name == self.external["building"]
                or name in (self._replacing(), self._mutating())
                or not (manage_data.vector_root() / name / "metadata.db").exists()):
            if self.files is not None:
                self._files_token += 1
                self.files = None
            return
        sig = manage_data.signature(name)
        if self.files and self.files["name"] == name and self.files["signature"] == sig:
            return
        for worker in list(self._workers):
            if isinstance(worker, FilesWorker):
                worker.requestInterruption()
        self._files_token += 1
        self.files = {"name": name, "signature": sig, "version": self._files_token, "status": "loading",
                      "rows": None, "payload": None, "error": None, "missing": None, "summary": {}}
        worker = FilesWorker(self._files_token, name)
        worker.loaded.connect(self._on_files)
        worker.checked.connect(self._on_missing)
        self._start(worker)

    def _on_files(self, token, loaded, error):
        if not self.files or token != self.files["version"]:
            return
        if loaded is None:
            self.files.update(status="error", error=error)
        else:
            self.files.update(status="ready", rows=loaded["rows"], payload=loaded["payload"], summary=loaded["summary"])
        self.schedule_push()

    def _on_missing(self, token, missing):
        if not self.files or token != self.files["version"]:
            return
        self.files["missing"] = missing
        self.schedule_push()

    def _status(self, entry, info):
        if entry["name"] == self.external["building"]:
            return "building"
        if not entry["folder"]:
            return "missing"
        if not entry["registered"]:
            return "leftover"
        if info is not None and not info["complete"]:
            return "incomplete"
        return "ready"

    def _entry_state(self, entry):
        cached = self.info.get(entry["name"])
        info = cached[1] if cached else None
        return {
            "name": entry["name"],
            "status": self._status(entry, info),
            "registered": entry["registered"],
            "folder": entry["folder"],
            "backup": entry["backup"],
            "backup_stale": entry["backup_stale"],
            "model": entry["model"],
            "chunk_size": entry["chunk_size"],
            "chunk_overlap": entry["chunk_overlap"],
            "info": info,
            "loading": entry["folder"] and info is None and entry["name"] != self.external["building"],
        }

    def _files_state(self):
        files = self.files
        if not files:
            return None
        rows = files["rows"]
        return {
            "name": files["name"],
            "version": files["version"],
            "status": files["status"],
            "error": files["error"],
            "count": len(rows) if rows is not None else None,
            "summary": files["summary"],
            "missing": len(files["missing"]) if files["missing"] is not None else None,
            "checked": files["missing"] is not None,
        }

    def blocked_reason(self):
        if self.task:
            return "Wait for the current task to finish."
        if self.external["tools_busy"]:
            return "A backup or restore is running on the Tools tab. Wait for it to finish."
        return None

    def build_state(self):
        return {
            "databases": [self._entry_state(entry) for entry in self.entries],
            "selected": self.selected,
            "files": self._files_state(),
            "task": self.task,
            "notice": self.notice,
            "blocked": self.blocked_reason(),
            "building": self.external["building"],
            "config_error": self.config_error,
            "kinds": [{"key": key, "label": label} for key, label, _ in manage_data.KINDS] + [{"key": "other", "label": "Other"}],
        }

    def database_task_running(self):
        return self.task is not None

    def busy_message(self):
        return TASK_MESSAGES.get(self.task["kind"]) if self.task else None

    def cleanup(self):
        self._poll.stop()
        for worker in list(self._workers):
            worker.requestInterruption()
        for worker in list(self._workers):
            worker.wait(10000 if isinstance(worker, (TaskWorker, DocumentMutationWorker, DocumentAdditionWorker)) else 3000)

    def js_select(self, name):
        if self._entry(name) is None:
            return {"error": "That database is no longer listed."}
        self.selected = name
        self._pending_add = None
        self._ensure_files()
        self.schedule_push()
        return {"ok": True}

    def js_files(self, name, version):
        files = self.files
        if not files or files["name"] != name or files["version"] != version or files["rows"] is None:
            return {"stale": True}
        return {"name": name, "version": version, "payload": files["payload"]}

    def js_missing(self, name, version):
        files = self.files
        if not files or files["name"] != name or files["version"] != version or files["missing"] is None:
            return {"stale": True}
        return {"missing": files["missing"]}

    def _row_path(self, version, index):
        files = self.files
        if not files or files["version"] != version or files["rows"] is None or not 0 <= index < len(files["rows"]):
            return None
        return files["rows"][index][1]

    def js_open_file(self, version, index):
        path = self._row_path(version, index)
        if path is None:
            return {"error": "The file list changed. Try again."}
        if not path or not os.path.exists(path):
            return {"missing": True, "path": path}
        open_file(path)
        return {"ok": True}

    def js_reveal_file(self, version, index):
        path = self._row_path(version, index)
        if path is None:
            return {"error": "The file list changed. Try again."}
        if path and os.path.exists(path):
            if sys.platform == "win32":
                subprocess.Popen(f'explorer /select,"{os.path.normpath(path)}"')
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))
            return {"ok": True}
        parent = Path(path).parent if path else None
        if parent and parent.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(parent)))
            return {"missing": True, "path": path, "opened_folder": True}
        return {"missing": True, "path": path}

    def js_remove_file(self, database_name, document_id):
        files = self.files
        if database_name != self.selected or not files or files["name"] != database_name or files["rows"] is None:
            return {"error": "The database changed. Select the document and try again."}
        row = next((item for item in files["rows"] if item[4] == document_id), None)
        if row is None:
            return {"error": "The file list changed. Select the document and try again."}
        reason = self.blocked_reason()
        if reason:
            return {"error": reason}
        name = self.selected
        entry = self._entry(name)
        if entry is None or not entry["folder"]:
            return {"error": "That database is no longer available."}
        document_name, _path, _chunks, _digest, _id, _copies, _group_chunks = row

        for worker in list(self._workers):
            if isinstance(worker, (FilesWorker, InfoWorker)):
                worker.requestInterruption()
        self._files_token += 1
        self.files = None
        self.task = {"kind": "remove_document", "name": name, "file": document_name}
        self.notice = None
        worker = DocumentMutationWorker(name, document_name, document_id)
        worker.done.connect(self._on_document_removed)
        self._start(worker)
        self.schedule_push()
        return {"ok": True}

    def _on_document_removed(self, database_name, document_name, result, error):
        self.task = None
        if error:
            self.notice = {
                "kind": "error",
                "message": f"{document_name} could not be removed from {database_name}: {error}",
            }
        else:
            removed = int((result or {}).get("removed_chunks", 0))
            suffix = f" ({removed} chunk{'s' if removed != 1 else ''} hidden from search)" if removed else ""
            self.notice = {
                "kind": "ok",
                "message": f"Removed {document_name} from {database_name}{suffix}. Rebuild the index later to reclaim its space.",
            }
        self.info.pop(database_name, None)
        self.files = None
        self.refresh()

    def _choose_add(self, name, folder):
        entry = self._entry(name)
        if name != self.selected or entry is None or not entry["folder"] or not entry["registered"]:
            return {"error": "Select an available database first."}
        reason = self.blocked_reason()
        if reason:
            return {"error": reason}
        if folder:
            chosen = QFileDialog.getExistingDirectory(self, "Choose folder to add", str(PROJECT_ROOT))
            if not chosen:
                return {"cancelled": True}
            paths = sorted(p for p in Path(chosen).rglob("*") if p.is_file())
        else:
            chosen = QFileDialog.getOpenFileNames(self, "Choose files to add", str(PROJECT_ROOT))[0]
            if not chosen:
                return {"cancelled": True}
            paths = [Path(p) for p in chosen]
        compatible = [str(p) for p in paths if p.suffix.lower() in ALLOWED_EXTENSIONS]
        skipped = [p.name for p in paths if p.suffix.lower() not in ALLOWED_EXTENSIONS]
        if not compatible:
            return {"error": "No supported files were selected."}
        self._pending_add = {"name": name, "paths": compatible, "skipped": skipped}
        return {"count": len(compatible), "skipped": skipped, "folder": folder}

    def js_choose_add_files(self, name):
        return self._choose_add(name, False)

    def js_choose_add_folder(self, name):
        return self._choose_add(name, True)

    def js_cancel_add(self):
        self._pending_add = None
        return {"ok": True}

    def js_add_documents(self, name):
        pending = self._pending_add
        if not pending or pending["name"] != name or name != self.selected:
            return {"error": "The selected files changed. Choose them again."}
        reason = self.blocked_reason()
        if reason:
            return {"error": reason}
        self._pending_add = None
        self._skipped_add = pending["skipped"]
        for worker in list(self._workers):
            if isinstance(worker, (FilesWorker, InfoWorker)):
                worker.requestInterruption()
        self._files_token += 1
        self.files = None
        self.task = {"kind": "add_documents", "name": name, "count": len(pending["paths"])}
        self.notice = None
        worker = DocumentAdditionWorker(name, pending["paths"])
        worker.progress.connect(self._on_add_progress)
        worker.done.connect(self._on_documents_added)
        self._start(worker)
        self.schedule_push()
        return {"ok": True}

    def _on_add_progress(self, stage):
        if self.task and self.task["kind"] == "add_documents":
            self.task["stage"] = stage
            self.schedule_push()

    def _on_documents_added(self, name, result, error):
        self.task = None
        if error:
            self.notice = {"kind": "error", "message": f"Documents could not be added to {name}: {error}"}
        else:
            added = result.get("added_documents", 0)
            skipped = self._skipped_add + result.get("skipped", []) + result.get("not_added", [])
            self.notice = {
                "kind": "ok" if added else "warn",
                "message": f"Added {added} document{'s' if added != 1 else ''} ({result.get('added_chunks', 0)} chunks) to {name}."
                           if added else f"No new documents were added to {name}.",
                "details": [f"Skipped: {item}" for item in skipped],
            }
        self.info.pop(name, None)
        self.files = None
        self._skipped_add = []
        self.refresh()

    def _begin(self, kind, name):
        entry = self._entry(name)
        if entry is None:
            return {"error": "That database is no longer listed."}
        reason = self.blocked_reason()
        if reason:
            return {"error": reason}
        if name == self.external["building"]:
            return {"error": "This database is being created on the Create Database tab."}
        if kind == "backup" and not (entry["folder"] and entry["registered"]):
            return {"error": "Only a complete database can be backed up."}
        if kind == "restore" and (entry["folder"] or not entry["backup"]):
            return {"error": "There is no backup copy to restore."}
        if kind == "rebuild":
            cached = self.info.get(name)
            if not (entry["folder"] and entry["registered"]) or not cached or not cached[1].get("changes"):
                return {"error": "This database has no changes, so its index doesn't need rebuilding."}
        readers = []
        if kind in ("delete", "rebuild"):
            for worker in list(self._workers):
                if isinstance(worker, (FilesWorker, InfoWorker)):
                    worker.requestInterruption()
                    readers.append(worker)
            if self.files and self.files["name"] == name:
                self._files_token += 1
                self.files = None
        self.task = {"kind": kind, "name": name, "registered": entry["registered"]}
        self.notice = None
        worker = TaskWorker(kind, name, readers)
        worker.done.connect(self._on_task_done)
        self._start(worker)
        self.schedule_push()
        return {"ok": True}

    def js_delete(self, name, next_name=None):
        self._next_selection = next_name
        return self._begin("delete", name)

    def js_backup(self, name):
        return self._begin("backup", name)

    def js_restore(self, name):
        return self._begin("restore", name)

    def js_rebuild(self, name):
        return self._begin("rebuild", name)

    def _on_task_done(self, kind, name, result, error):
        registered = bool(self.task and self.task.get("registered"))
        self.task = None
        if error and kind == "rebuild":
            self.notice = {"kind": "error", "message": f"The index of {name} could not be rebuilt: {error}"}
        elif error:
            verb = {"delete": "deleted", "backup": "backed up", "restore": "restored"}[kind]
            self.notice = {"kind": "error", "message": f"{name} could not be {verb}: {error}"}
        elif kind == "delete":
            failed = (result or {}).get("failed") or []
            if failed:
                message = (f"{name} was removed from the list, but some of its files could not be deleted." if registered
                           else f"Some of the files in {name} could not be deleted.")
                self.notice = {"kind": "warn", "message": message, "details": failed}
            else:
                self.notice = {"kind": "ok", "message": f"Deleted {name}."}
                if self.selected == name:
                    self.selected = self._next_selection
        elif kind == "backup":
            self.notice = {"kind": "ok", "message": f"Backed up {name}."}
        elif kind == "rebuild":
            self.notice = {"kind": "ok", "message": f"Rebuilt the index of {name}."}
        else:
            self.notice = {"kind": "ok", "message": f"Restored {name} from its backup copy."}
        self.refresh()

    def js_dismiss_notice(self):
        self.notice = None
        self.schedule_push()

    def js_open_tab(self, name):
        widget = self.parentWidget()
        while widget is not None and not isinstance(widget, QTabWidget):
            widget = widget.parentWidget()
        if widget is None:
            return {"error": "Tab not found."}
        for index in range(widget.count()):
            if widget.tabText(index) == name:
                widget.setCurrentIndex(index)
                return {"ok": True}
        return {"error": "Tab not found."}

    def js_query_database(self, name):
        result = self.js_open_tab("Query Database")
        for widget in QApplication.allWidgets():
            select = getattr(widget, "select_database", None)
            if callable(select):
                select(name)
                break
        return result
