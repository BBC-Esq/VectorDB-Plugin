import os
import platform
import shutil
import subprocess
import time

from PySide6.QtCore import QObject, QSettings, QThread, Signal
from PySide6.QtWidgets import QMessageBox

from core.constants import PROJECT_ROOT, scrape_documentation
from modules.scraper import ScraperRegistry, ScraperWorker

MAX_CONCURRENT_SCRAPES = 6

QSETTINGS_ORG = "VectorDB-Plugin"
QSETTINGS_APP = "ScrapeDocumentation"
RATE_LIMITED_KEY = "rate_limited_scrapes"


def _load_rate_limited_set() -> set[str]:
    s = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    val = s.value(RATE_LIMITED_KEY, [])
    if isinstance(val, str):
        val = [val] if val else []
    if val is None:
        val = []
    return {str(v) for v in val}


def _save_rate_limited_set(names: set[str]) -> None:
    s = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
    s.setValue(RATE_LIMITED_KEY, sorted(names))


def _mark_rate_limited_persistent(name: str) -> None:
    names = _load_rate_limited_set()
    names.add(name)
    _save_rate_limited_set(names)


def _clear_rate_limited_persistent(name: str) -> None:
    names = _load_rate_limited_set()
    names.discard(name)
    _save_rate_limited_set(names)


def scraped_root():
    return os.path.join(str(PROJECT_ROOT), "Scraped_Documentation")


def folder_for(doc_name):
    return os.path.join(scraped_root(), scrape_documentation[doc_name]["folder"])


def count_pages(folder_path):
    try:
        if os.path.exists(folder_path):
            return len([f for f in os.listdir(folder_path) if f.endswith(".html")])
    except Exception:
        pass
    return 0


class ScrapeTool(QObject):
    changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.active: dict[str, dict] = {}
        self.rows: dict[str, dict] = {}
        self.scraped: set[str] = set()
        names = self.doc_names()
        self.selected = names[0] if names else None
        self._restore_rate_limited_rows()
        self.refresh()

    def doc_names(self):
        return sorted(scrape_documentation.keys(), key=str.lower)

    def refresh(self):
        root = scraped_root()
        self.scraped = {
            name for name in self.doc_names()
            if os.path.exists(os.path.join(root, scrape_documentation[name]["folder"]))
        }

    def _restore_rate_limited_rows(self):
        for doc_name in sorted(_load_rate_limited_set()):
            doc_info = scrape_documentation.get(doc_name)
            if not doc_info or "folder" not in doc_info:
                _clear_rate_limited_persistent(doc_name)
                continue
            folder_path = folder_for(doc_name)
            if not os.path.exists(folder_path):
                _clear_rate_limited_persistent(doc_name)
                continue
            self.rows[doc_name] = {"status": "rate_limited", "pages": count_pages(folder_path), "folder": folder_path}

    def state(self):
        docs = [{"name": name, "scraped": name in self.scraped} for name in self.doc_names()]
        rows = [
            {"name": name, "status": row["status"], "pages": row["pages"],
             "running": name in self.active, "started": row.get("started")}
            for name, row in self.rows.items()
        ]
        return {
            "docs": docs,
            "selected": self.selected,
            "active": len(self.active),
            "limit": MAX_CONCURRENT_SCRAPES,
            "rows": rows,
        }

    def select(self, name):
        if name in scrape_documentation:
            self.selected = name
            self.changed.emit()

    def make_worker(self, url, folder, scraper_class, name, resume):
        return ScraperWorker(url, folder, scraper_class, name=name, resume=resume)

    def start(self, parent_widget, name):
        doc_info = scrape_documentation.get(name)
        if not doc_info or "URL" not in doc_info or "folder" not in doc_info:
            QMessageBox.critical(parent_widget, "Error", "Incomplete configuration for the selection.")
            return
        if name in self.active:
            QMessageBox.information(parent_widget, "Already Scraping", f"'{name}' is already being scraped.")
            return
        if len(self.active) >= MAX_CONCURRENT_SCRAPES:
            QMessageBox.warning(
                parent_widget,
                "Concurrent Scrape Limit Reached",
                f"You can run at most {MAX_CONCURRENT_SCRAPES} scrapes at the same time. "
                f"Wait for one to finish (or cancel one) before starting another.",
            )
            return

        url = doc_info["URL"]
        folder = doc_info["folder"]
        scraper_class = ScraperRegistry.get_scraper(doc_info.get("scraper_class", "BaseScraper"))
        folder_path = folder_for(name)

        resume = False
        if os.path.exists(folder_path):
            msg_box = QMessageBox(
                QMessageBox.Warning,
                "Existing Folder",
                f"A scrape folder already exists for {name}.",
                QMessageBox.NoButton,
                parent_widget,
            )
            msg_box.setInformativeText(
                "Resume: pick up where the last run left off (already-saved pages are skipped; "
                "queued and failed URLs are retried).\n\n"
                "Start Fresh: delete the existing folder contents and re-scrape from scratch.\n\n"
                "Cancel: do nothing."
            )
            resume_btn = msg_box.addButton("Resume", QMessageBox.AcceptRole)
            fresh_btn = msg_box.addButton("Start Fresh", QMessageBox.DestructiveRole)
            cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            msg_box.setDefaultButton(resume_btn)
            msg_box.exec()
            clicked = msg_box.clickedButton()
            if clicked is None or clicked == cancel_btn:
                return
            resume = clicked == resume_btn
            if not resume:
                _clear_rate_limited_persistent(name)
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception:
                        pass

        self.rows.pop(name, None)
        self.rows[name] = {"status": "starting", "pages": 0, "folder": folder_path, "started": time.time()}

        worker = self.make_worker(url, folder, scraper_class, name, resume)
        self.refresh()
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status_updated.connect(self._on_status)
        worker.scraping_finished.connect(self._on_worker_finished)
        worker.scraping_finished.connect(thread.quit)
        thread.finished.connect(lambda n=name: self._on_thread_finished(n))
        thread.finished.connect(thread.deleteLater)
        self.active[name] = {"worker": worker, "thread": thread}
        thread.start()
        self.changed.emit()

    def _on_status(self, doc_name, status):
        row = self.rows.get(doc_name)
        if row is None or doc_name not in self.active:
            return
        try:
            row["pages"] = int(status)
        except ValueError:
            pass
        if row["status"] == "starting":
            row["status"] = "scraping"
        self.changed.emit()

    def _on_worker_finished(self, doc_name, was_cancelled, was_rate_limited):
        row = self.rows.get(doc_name)
        if row is None:
            return
        row["pages"] = count_pages(row["folder"])
        if was_cancelled:
            row["status"] = "cancelled"
        elif was_rate_limited:
            row["status"] = "rate_limited"
            _mark_rate_limited_persistent(doc_name)
        else:
            row["status"] = "completed"
            _clear_rate_limited_persistent(doc_name)
        row["finished"] = time.time()
        self.selected = doc_name
        self.refresh()
        self.changed.emit()

    def _on_thread_finished(self, doc_name):
        self.active.pop(doc_name, None)
        self.changed.emit()

    def cancel(self, name):
        entry = self.active.get(name)
        if not entry:
            return
        row = self.rows.get(name)
        if row is not None and row["status"] in ("starting", "scraping"):
            row["status"] = "cancelling"
        try:
            entry["worker"].cancel()
        except Exception as e:
            print(f"Error cancelling {name}: {e}")
        self.changed.emit()

    def dismiss(self, name):
        if name in self.active:
            return
        row = self.rows.pop(name, None)
        if row is not None and row["status"] == "rate_limited":
            _clear_rate_limited_persistent(name)
        self.changed.emit()

    def open_folder(self, parent_widget, name):
        folder_path = self.rows[name]["folder"] if name in self.rows else folder_for(name)
        if not os.path.exists(folder_path):
            QMessageBox.information(parent_widget, "Folder Not Found", "The folder hasn't been created yet (no pages scraped).")
            return
        system = platform.system()
        if system == "Windows":
            os.startfile(folder_path)
        elif system == "Darwin":
            subprocess.Popen(["open", folder_path])
        else:
            subprocess.Popen(["xdg-open", folder_path])

    def cleanup(self):
        for entry in list(self.active.values()):
            worker = entry.get("worker")
            thread = entry.get("thread")
            if worker is not None:
                try:
                    worker.cancel()
                except Exception:
                    pass
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(5000)
