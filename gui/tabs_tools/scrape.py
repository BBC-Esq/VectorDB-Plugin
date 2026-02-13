import os
import platform
import shutil
import subprocess

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QMessageBox,
)

from module_scraper import ScraperRegistry, ScraperWorker
from constants import scrape_documentation


class ScrapeDocumentationTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setToolTip(
            "Tab for scraping documentation from the selected source."
        )
        self.init_ui()

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

        self.status_label = QLabel()
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setOpenExternalLinks(False)
        self.status_label.linkActivated.connect(self.open_folder)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.status_label.setText(
            '<span style="color:#4CAF50;"><b>Pages scraped:</b></span> 0'
        )
        main_layout.addWidget(self.status_label)

        self.current_folder = ""
        self.current_doc_name = ""

    def populate_combo_box(self) -> None:
        doc_options = sorted(scrape_documentation.keys(), key=str.lower)
        model = QStandardItemModel()

        scraped_dir = os.path.join(
            os.path.dirname(__file__),
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

        url = doc_info["URL"]
        folder = doc_info["folder"]
        scraper_name = doc_info.get("scraper_class", "BaseScraper")
        scraper_class = ScraperRegistry.get_scraper(scraper_name)

        self.current_folder = os.path.join(
            os.path.dirname(__file__),
            "Scraped_Documentation",
            folder,
        )
        self.current_doc_name = selected_doc

        if os.path.exists(self.current_folder):
            msg_box = QMessageBox(
                QMessageBox.Warning,
                "Existing Folder Warning",
                f"Folder already exists for {selected_doc}",
                QMessageBox.Ok | QMessageBox.Cancel,
                self,
            )
            msg_box.setInformativeText(
                "Proceeding will delete its contents and start over."
            )
            msg_box.setDefaultButton(QMessageBox.Cancel)

            if msg_box.exec() == QMessageBox.Cancel:
                self.scrape_button.setEnabled(True)
                return

            for filename in os.listdir(self.current_folder):
                file_path = os.path.join(self.current_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception:
                    pass

        self.status_label.setText(
            f'<span style="color:#FF9800;"><b>Scraping '
            f'{self.current_doc_name}...</b></span> '
            f'<span style="color:#4CAF50;"><b>Pages scraped:</b></span> 0 '
            f'<span style="color:#2196F3;"><a href="open_folder" '
            f'style="color:#2196F3;">Open Folder</a></span>'
        )
        self.scrape_button.setEnabled(False)

        self.worker = ScraperWorker(url, folder, scraper_class)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status_updated.connect(self.update_status)
        self.worker.scraping_finished.connect(self.scraping_finished)
        self.worker.scraping_finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def update_status(self, status: str) -> None:
        self.status_label.setText(
            f'<span style="color:#FF9800;"><b>Scraping '
            f'{self.current_doc_name}...</b></span> '
            f'<span style="color:#4CAF50;"><b>Pages scraped:</b></span> '
            f'{status} '
            f'<span style="color:#2196F3;"><a href="open_folder" '
            f'style="color:#2196F3;">Open Folder</a></span>'
        )

    def scraping_finished(self) -> None:
        self.scrape_button.setEnabled(True)
        final_count = len(
            [
                f
                for f in os.listdir(self.current_folder)
                if f.endswith(".html")
            ]
        )
        self.status_label.setText(
            f'<span style="color:#FF9800;"><b>Scraping '
            f'{self.current_doc_name} completed.</b></span> '
            f'<span style="color:#4CAF50;"><b>Pages scraped:</b></span> '
            f'{final_count} '
            f'<span style="color:#2196F3;"><a href="open_folder" '
            f'style="color:#2196F3;">Open Folder</a></span>'
        )
        self.populate_combo_box()
        idx = self.doc_combo.findText(self.current_doc_name)
        if idx >= 0:
            self.doc_combo.setCurrentIndex(idx)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def open_folder(self, link: str) -> None:
        if link == "open_folder":
            system = platform.system()
            if system == "Windows":
                os.startfile(self.current_folder)
            elif system == "Darwin":          # macOS
                subprocess.Popen(["open", self.current_folder])
            else:                             # Linux / BSD
                subprocess.Popen(["xdg-open", self.current_folder])
