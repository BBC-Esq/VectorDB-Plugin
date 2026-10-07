import threading
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox

from gui.download_model import ModelDownloader, model_downloaded_signal
from gui.tabs_models import catalog
from gui.web_common.web_tab import WebTab

WEB_PAGE = Path(__file__).resolve().parent / "web" / "models_tab.html"


class VectorModelsTab(WebTab):
    download_finished = Signal(str)

    def __init__(self, parent=None):
        super().__init__(WEB_PAGE, parent)
        (Path("Models") / "vector").mkdir(parents=True, exist_ok=True)
        self._downloading = None
        model_downloaded_signal.downloaded.connect(self._on_model_downloaded)
        model_downloaded_signal.failed.connect(self._on_download_failed)
        self.download_finished.connect(self._on_download_finished)

    def build_state(self):
        return catalog.build_payload(self._downloading)

    def js_download(self, repo_id):
        if self._downloading:
            return
        model_info = catalog.find_model(repo_id)
        if model_info is None:
            return
        if catalog.is_downloaded(model_info):
            reply = QMessageBox.question(
                self,
                "Model Already Downloaded",
                f"'{model_info['name']}' is already downloaded.\n\nRe-download it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        self._downloading = repo_id
        self.push_state()
        model_downloader = ModelDownloader(model_info, model_info["type"])
        threading.Thread(target=self._run_download, args=(model_downloader, repo_id), daemon=True).start()

    def _run_download(self, model_downloader, repo_id):
        try:
            model_downloader.download()
        finally:
            self.download_finished.emit(repo_id)

    def _on_model_downloaded(self, model_name, model_type):
        if model_type == "vector":
            self.push_state()

    def _on_download_failed(self, message):
        QMessageBox.critical(self, "Download Failed", message)

    def _on_download_finished(self, repo_id):
        if self._downloading == repo_id:
            self._downloading = None
        self.push_state()
