import json
import logging
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QUrl, Slot
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from core.utilities import theme_manager
from gui.web_common.theme import theme_payload


class WebPage(QWebEnginePage):
    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
            return False
        return super().acceptNavigationRequest(url, navigation_type, is_main_frame)

    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            logging.error("%s: %s (line %s)", Path(QUrl(source).path()).name, message, line)


class WebBridge(QObject):
    def __init__(self, tab):
        super().__init__(tab)
        self._tab = tab

    @Slot(str, str, result=str)
    def call(self, name, args_json):
        handler = getattr(self._tab, f"js_{name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown action: {name}"})
        try:
            result = handler(**json.loads(args_json or "{}"))
        except Exception as e:
            logging.exception("The %s action failed", name)
            return json.dumps({"error": str(e)})
        return json.dumps({} if result is None else result)


class WebTab(QWidget):
    def __init__(self, page_path, parent=None):
        super().__init__(parent)
        self._ready = False
        self._sent_state = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view = QWebEngineView(self)
        self._view.setContextMenuPolicy(Qt.NoContextMenu)
        self._page = WebPage(QWebEngineProfile.defaultProfile(), self._view)
        settings = self._page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, False)
        self._page.setBackgroundColor(QColor(theme_payload(theme_manager.current)["colors"]["bg_window"]))
        self._channel = QWebChannel(self._page)
        self._channel.registerObject("bridge", WebBridge(self))
        self._page.setWebChannel(self._channel)
        self._page.loadFinished.connect(self._on_load_finished)
        self._view.setPage(self._page)
        layout.addWidget(self._view)
        theme_manager.changed.connect(self._apply_theme)
        self._view.load(QUrl.fromLocalFile(str(page_path)))

    def build_state(self):
        raise NotImplementedError

    def showEvent(self, event):
        super().showEvent(event)
        self.push_state()

    def push_state(self):
        if not self._ready:
            return
        state = json.dumps(self.build_state())
        if state != self._sent_state:
            self._sent_state = state
            self._page.runJavaScript(f"TabApp.setState({state});")

    def remember_state(self, state):
        self._sent_state = json.dumps(state)

    def _on_load_finished(self, ok):
        if not ok:
            if not self._ready:
                logging.error("%s failed to load", type(self).__name__)
            return
        self._ready = True
        self._sent_state = json.dumps(self.build_state())
        theme = json.dumps(theme_payload(theme_manager.current))
        self._page.runJavaScript(f"VDB.applyTheme({theme}); TabApp.init({self._sent_state});")

    def _apply_theme(self, theme_name):
        theme = theme_payload(theme_name)
        self._page.setBackgroundColor(QColor(theme["colors"]["bg_window"]))
        if self._ready:
            self._page.runJavaScript(f"VDB.applyTheme({json.dumps(theme)});")
