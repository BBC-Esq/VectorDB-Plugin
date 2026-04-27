import os
import platform
import shutil
import subprocess

from PySide6.QtCore import Qt, QThread, QSettings
from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
)

from modules.scraper import ScraperRegistry, ScraperWorker
from core.constants import scrape_documentation, PROJECT_ROOT


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


class ScrapeRowWidget(QWidget):
    """One row in the active-scrapes list. Owns the per-scrape Cancel/Open buttons."""

    def __init__(self, doc_name: str, folder_path: str, on_cancel, on_open):
        super().__init__()
        self.doc_name = doc_name
        self.folder_path = folder_path
        self._on_cancel = on_cancel
        self._on_open = on_open

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        self.label = QLabel()
        self.label.setTextFormat(Qt.RichText)
        self._set_label("Starting...", count=0, color="#FF9800")
        layout.addWidget(self.label, 1)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_clicked)
        layout.addWidget(self.cancel_btn)

        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self._open_clicked)
        layout.addWidget(self.open_btn)

    def _set_label(self, status_text: str, count: int, color: str):
        self.label.setText(
            f'<span style="color:#4CAF50;"><b>{self.doc_name}</b></span> '
            f'<span style="color:{color};">{status_text}</span> '
            f'<span style="color:#4CAF50;">Pages scraped:</span> {count}'
        )

    def update_count(self, count: int):
        self._set_label("Scraping...", count=count, color="#FF9800")

    def mark_completed(self, count: int):
        self._set_label("Completed.", count=count, color="#4CAF50")
        self.cancel_btn.setEnabled(False)

    def mark_cancelled(self, count: int):
        self._set_label("Cancelled.", count=count, color="#9E9E9E")
        self.cancel_btn.setEnabled(False)

    def mark_rate_limited(self, count: int):
        self._set_label(
            "Rate-limited - partial state saved. Click 'Scrape' again and choose Resume.",
            count=count, color="#FFC107",
        )
        self.cancel_btn.setEnabled(False)

    def _cancel_clicked(self):
        self.cancel_btn.setEnabled(False)
        self._set_label("Cancelling...", count=self._current_count(), color="#9E9E9E")
        self._on_cancel(self.doc_name)

    def _open_clicked(self):
        self._on_open(self.folder_path)

    def _current_count(self) -> int:
        try:
            if os.path.exists(self.folder_path):
                return len([f for f in os.listdir(self.folder_path) if f.endswith(".html")])
        except Exception:
            pass
        return 0


class ScrapeDocumentationTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setToolTip(
            "Tab for scraping documentation from the selected source."
        )
        self.active_workers: dict[str, dict] = {}
        self.restored_rate_limited_rows: dict[str, QListWidgetItem] = {}
        self.init_ui()
        self._restore_rate_limited_rows()

    def init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        label = QLabel("Select Documentation:")
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(label)

        hbox = QHBoxLayout()
        self.doc_combo = QComboBox()
        self.populate_combo_box()
        hbox.addWidget(self.doc_combo)

        self.scrape_button = QPushButton("Scrape")
        self.scrape_button.clicked.connect(self.start_scraping)
        hbox.addWidget(self.scrape_button)

        hbox.setStretch(0, 1)
        hbox.setStretch(1, 1)
        main_layout.addLayout(hbox)

        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._refresh_summary()
        main_layout.addWidget(self.summary_label)

        self.scrape_list = QListWidget()
        self.scrape_list.setSelectionMode(QListWidget.NoSelection)
        main_layout.addWidget(self.scrape_list, 1)

    def _refresh_summary(self) -> None:
        n = len(self.active_workers)
        self.summary_label.setText(
            f'<span style="color:#2196F3;"><b>Active scrapes:</b></span> '
            f'{n} / {MAX_CONCURRENT_SCRAPES}'
        )

    def _restore_rate_limited_rows(self) -> None:
        persisted = _load_rate_limited_set()
        if not persisted:
            return
        scraped_dir = os.path.join(str(PROJECT_ROOT), "Scraped_Documentation")
        for doc_name in sorted(persisted):
            doc_info = scrape_documentation.get(doc_name)
            if not doc_info or "folder" not in doc_info:
                _clear_rate_limited_persistent(doc_name)
                continue
            folder_path = os.path.join(scraped_dir, doc_info["folder"])
            if not os.path.exists(folder_path):
                _clear_rate_limited_persistent(doc_name)
                continue
            count = 0
            try:
                count = len([f for f in os.listdir(folder_path) if f.endswith(".html")])
            except Exception:
                pass
            row = ScrapeRowWidget(
                doc_name=doc_name,
                folder_path=folder_path,
                on_cancel=lambda _n: None,
                on_open=self.open_folder,
            )
            row.mark_rate_limited(count)
            item = QListWidgetItem(self.scrape_list)
            item.setSizeHint(row.sizeHint())
            self.scrape_list.addItem(item)
            self.scrape_list.setItemWidget(item, row)
            self.restored_rate_limited_rows[doc_name] = item

    def populate_combo_box(self) -> None:
        doc_options = sorted(scrape_documentation.keys(), key=str.lower)
        model = QStandardItemModel()

        scraped_dir = os.path.join(
            str(PROJECT_ROOT),
            "Scraped_Documentation",
        )

        for doc in doc_options:
            folder = scrape_documentation[doc]["folder"]
            folder_path = os.path.join(scraped_dir, folder)
            item = QStandardItem(doc)
            if os.path.exists(folder_path):
                item.setForeground(QColor("#e75959"))
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            model.appendRow(item)

        self.doc_combo.setModel(model)

    def start_scraping(self) -> None:
        selected_doc = self.doc_combo.currentText()
        doc_info = scrape_documentation.get(selected_doc)
        if not doc_info or "URL" not in doc_info or "folder" not in doc_info:
            self.show_error("Incomplete configuration for the selection.")
            return

        if selected_doc in self.active_workers:
            QMessageBox.information(
                self,
                "Already Scraping",
                f"'{selected_doc}' is already being scraped.",
            )
            return

        if len(self.active_workers) >= MAX_CONCURRENT_SCRAPES:
            QMessageBox.warning(
                self,
                "Concurrent Scrape Limit Reached",
                f"You can run at most {MAX_CONCURRENT_SCRAPES} scrapes at the same time. "
                f"Wait for one to finish (or cancel one) before starting another.",
            )
            return

        url = doc_info["URL"]
        folder = doc_info["folder"]
        scraper_name = doc_info.get("scraper_class", "BaseScraper")
        scraper_class = ScraperRegistry.get_scraper(scraper_name)

        folder_path = os.path.join(
            str(PROJECT_ROOT),
            "Scraped_Documentation",
            folder,
        )

        resume = False
        if os.path.exists(folder_path):
            msg_box = QMessageBox(
                QMessageBox.Warning,
                "Existing Folder",
                f"A scrape folder already exists for {selected_doc}.",
                QMessageBox.NoButton,
                self,
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
            resume = (clicked == resume_btn)
            if not resume:
                _clear_rate_limited_persistent(selected_doc)
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception:
                        pass

        if selected_doc in self.restored_rate_limited_rows:
            old_item = self.restored_rate_limited_rows.pop(selected_doc)
            old_row = self.scrape_list.row(old_item)
            if old_row >= 0:
                self.scrape_list.takeItem(old_row)

        row = ScrapeRowWidget(
            doc_name=selected_doc,
            folder_path=folder_path,
            on_cancel=self.cancel_scrape,
            on_open=self.open_folder,
        )
        item = QListWidgetItem(self.scrape_list)
        item.setSizeHint(row.sizeHint())
        self.scrape_list.addItem(item)
        self.scrape_list.setItemWidget(item, row)

        worker = ScraperWorker(url, folder, scraper_class, name=selected_doc, resume=resume)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status_updated.connect(self.update_status)
        # Phase 1: worker emits → update the row UI + ask the thread to quit.
        # We must NOT drop our Python references to worker/thread here, because
        # thread.quit() takes a moment to wind down the event loop.
        worker.scraping_finished.connect(self._on_worker_finished)
        worker.scraping_finished.connect(thread.quit)
        # Phase 2: thread truly exited — safe to release references.
        thread.finished.connect(lambda n=selected_doc: self._on_thread_finished(n))
        thread.finished.connect(thread.deleteLater)

        self.active_workers[selected_doc] = {
            "worker": worker,
            "thread": thread,
            "row": row,
            "item": item,
            "folder_path": folder_path,
        }

        thread.start()
        self._refresh_summary()

    def update_status(self, doc_name: str, status: str) -> None:
        entry = self.active_workers.get(doc_name)
        if not entry:
            return
        try:
            count = int(status)
        except ValueError:
            count = 0
        entry["row"].update_count(count)

    def _on_worker_finished(self, doc_name: str, was_cancelled: bool, was_rate_limited: bool) -> None:
        """Phase 1: worker emitted scraping_finished. Update the row UI but do NOT
        drop Python references — thread.quit() still has to wind down."""
        entry = self.active_workers.get(doc_name)
        if not entry:
            return
        row = entry["row"]
        folder_path = entry["folder_path"]
        count = 0
        try:
            if os.path.exists(folder_path):
                count = len([f for f in os.listdir(folder_path) if f.endswith(".html")])
        except Exception:
            pass
        if was_cancelled:
            row.mark_cancelled(count)
        elif was_rate_limited:
            row.mark_rate_limited(count)
            _mark_rate_limited_persistent(doc_name)
        else:
            row.mark_completed(count)
            _clear_rate_limited_persistent(doc_name)

        self.populate_combo_box()
        idx = self.doc_combo.findText(doc_name)
        if idx >= 0:
            self.doc_combo.setCurrentIndex(idx)

    def _on_thread_finished(self, doc_name: str) -> None:
        """Phase 2: thread event loop has exited. Now safe to release refs."""
        self.active_workers.pop(doc_name, None)
        self._refresh_summary()

    def cancel_scrape(self, doc_name: str) -> None:
        entry = self.active_workers.get(doc_name)
        if not entry:
            return
        try:
            entry["worker"].cancel()
        except Exception as e:
            print(f"Error cancelling {doc_name}: {e}")

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def open_folder(self, folder_path: str) -> None:
        if not os.path.exists(folder_path):
            QMessageBox.information(
                self, "Folder Not Found",
                "The folder hasn't been created yet (no pages scraped).",
            )
            return
        system = platform.system()
        if system == "Windows":
            os.startfile(folder_path)
        elif system == "Darwin":
            subprocess.Popen(["open", folder_path])
        else:
            subprocess.Popen(["xdg-open", folder_path])
