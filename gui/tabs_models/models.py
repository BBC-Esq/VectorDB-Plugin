import json
import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QWidget

from core.utilities import theme_manager
from gui.download_model import ModelDownloader, model_downloaded_signal
from gui.tabs_models import catalog

WEB_PAGE = Path(__file__).resolve().parent / "web" / "models_tab.html"


class ModelsBridge(QObject):
    download_requested = Signal(str)

    @Slot(str)
    def requestDownload(self, repo_id):
        self.download_requested.emit(repo_id)


class ModelsPage(QWebEnginePage):
    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
            return False
        return super().acceptNavigationRequest(url, navigation_type, is_main_frame)

    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            logging.error("Models tab page error: %s (line %s)", message, line)


class VectorModelsTab(QWidget):
    download_finished = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        (Path("Models") / "vector").mkdir(parents=True, exist_ok=True)
        self._ready = False
        self._downloading = None
        self._sent_payload = None
        theme_manager.changed.connect(self._apply_theme)
        model_downloaded_signal.downloaded.connect(self._on_model_downloaded)
        model_downloaded_signal.failed.connect(self._on_download_failed)
        self.download_finished.connect(self._on_download_finished)
        self._create_view()

    def showEvent(self, event):
        super().showEvent(event)
        self._push_data()

    def _create_view(self):
        self._view = QWebEngineView(self)
        self._view.setContextMenuPolicy(Qt.NoContextMenu)
        self._profile = QWebEngineProfile(self)
        self._page = ModelsPage(self._profile, self._view)
        settings = self._page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, False)
        self._page.setBackgroundColor(QColor(catalog.theme_payload(theme_manager.current)["colors"]["bg_window"]))
        self._bridge = ModelsBridge(self)
        self._bridge.download_requested.connect(self._on_download_requested)
        self._channel = QWebChannel(self._page)
        self._channel.registerObject("bridge", self._bridge)
        self._page.setWebChannel(self._channel)
        self._page.loadFinished.connect(self._on_load_finished)
        self._view.setPage(self._page)
        self._layout.addWidget(self._view)
        self._view.load(QUrl.fromLocalFile(str(WEB_PAGE)))

    def _on_load_finished(self, ok):
        if not ok:
            if not self._ready:
                logging.error("The Models tab page failed to load: %s", WEB_PAGE)
            return
        self._ready = True
        self._sent_payload = json.dumps(catalog.build_payload(self._downloading))
        theme = json.dumps(catalog.theme_payload(theme_manager.current))
        self._page.runJavaScript(f"ModelsTab.init({self._sent_payload}, {theme});")

    def _push_data(self):
        if not self._ready:
            return
        payload = json.dumps(catalog.build_payload(self._downloading))
        if payload != self._sent_payload:
            self._sent_payload = payload
            self._page.runJavaScript(f"ModelsTab.setData({payload});")

    def _apply_theme(self, theme_name):
        theme = catalog.theme_payload(theme_name)
        self._page.setBackgroundColor(QColor(theme["colors"]["bg_window"]))
        if self._ready:
            self._page.runJavaScript(f"ModelsTab.applyTheme({json.dumps(theme)});")

    def _on_download_requested(self, repo_id):
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
        self._push_data()
        model_downloader = ModelDownloader(model_info, model_info["type"])
        threading.Thread(target=self._run_download, args=(model_downloader, repo_id), daemon=True).start()

    def _run_download(self, model_downloader, repo_id):
        try:
            model_downloader.download()
        finally:
            self.download_finished.emit(repo_id)

    def _on_model_downloaded(self, model_name, model_type):
        if model_type == "vector":
            self._push_data()

    def _on_download_failed(self, message):
        QMessageBox.critical(self, "Download Failed", message)

    def _on_download_finished(self, repo_id):
        if self._downloading == repo_id:
            self._downloading = None
        self._push_data()
