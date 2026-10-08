import threading
from ctypes import byref, c_int, sizeof, windll
from ctypes.wintypes import BOOL, DWORD, HWND
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QVBoxLayout

from gui.web_common.web_tab import WebTab
from gui.dialogs import ai_backends_data as data

WEB_PAGE = Path(__file__).resolve().parent / "web" / "ai_backends.html"


class ChatBackendsPage(WebTab):
    def __init__(self, selected="chatgpt", parent=None):
        data.normalize_config()
        self.selected = selected if selected in data.BACKENDS else data.BACKENDS[0]
        self.tests = {}
        self._runs = {}
        self._generation = {}
        self.on_close = None
        super().__init__(WEB_PAGE, parent)
        self._collector = QTimer(self)
        self._collector.setInterval(100)
        self._collector.timeout.connect(self._collect)

    def build_state(self):
        return data.build_state(self.selected, self.tests)

    def _state_reply(self, result):
        state = self.build_state()
        self.remember_state(state)
        result["state"] = state
        return result

    def js_select(self, backend):
        if backend in data.BACKENDS:
            self.selected = backend
            self.remember_state(self.build_state())
        return {"ok": True}

    def js_apply(self, key, value):
        result = data.apply_change(key, value)
        if result.get("ok") and result.get("changed"):
            self._forget_test(data.SETTING_BACKEND.get(key))
        return self._state_reply(result)

    def js_reveal(self, backend):
        try:
            return {"key": data.saved_key(backend)}
        except data.ConfigError as e:
            return {"error": str(e)}

    def js_check(self, backend):
        if backend not in data.TESTABLE:
            return {"error": "That backend can't be tested."}
        if backend in self._runs:
            return self._state_reply({"ok": True})
        try:
            cfg = data.load_config()
        except data.ConfigError as e:
            self.tests[backend] = {"status": "error", "message": str(e)}
            return self._state_reply({"ok": True})
        generation = self._generation.get(backend, 0)
        box = {}

        def run():
            try:
                box["result"] = data.check_connection(backend, cfg)
            except Exception as e:
                box["result"] = {"status": "error", "message": f"The test failed: {e}"}

        self._runs[backend] = (generation, box)
        self.tests[backend] = {"status": "running"}
        threading.Thread(target=run, name=f"check-{backend}", daemon=True).start()
        self._collector.start()
        return self._state_reply({"ok": True})

    def _forget_test(self, backend):
        if backend is None:
            return
        self._generation[backend] = self._generation.get(backend, 0) + 1
        self._runs.pop(backend, None)
        self.tests.pop(backend, None)

    def _collect(self):
        changed = False
        for backend, (generation, box) in list(self._runs.items()):
            if "result" not in box:
                continue
            del self._runs[backend]
            if generation == self._generation.get(backend, 0):
                self.tests[backend] = box["result"]
                changed = True
        if not self._runs:
            self._collector.stop()
        if changed:
            self.schedule_push()

    def js_close(self):
        if self.on_close:
            QTimer.singleShot(0, self.on_close)
        return {"ok": True}

    def cleanup(self):
        self._collector.stop()
        self._runs.clear()


class AIBackendsDialog(QDialog):
    def __init__(self, parent=None, initial_tab=0):
        super().__init__(parent)
        self.setWindowTitle("Chat Backend Settings")
        self.resize(780, 560)
        self.setMinimumSize(620, 460)
        selected = data.BACKENDS[initial_tab] if 0 <= initial_tab < len(data.BACKENDS) else data.BACKENDS[0]
        self.page = ChatBackendsPage(selected, self)
        self.page.on_close = self.finish_close
        self._closing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.page)

    def showEvent(self, event):
        super().showEvent(event)
        set_window_attribute = windll.dwmapi.DwmSetWindowAttribute
        hwnd = HWND(int(self.winId()))
        dark = BOOL(True)
        set_window_attribute(hwnd, DWORD(20), byref(dark), sizeof(dark))
        black = c_int(0xFF000000)
        set_window_attribute(hwnd, DWORD(34), byref(black), sizeof(black))

    def reject(self):
        if self._closing:
            return
        if not self.page._ready:
            self.finish_close()
            return
        self.page._page.runJavaScript("TabApp.requestClose()")
        QTimer.singleShot(2000, self.finish_close)

    def finish_close(self):
        if self._closing:
            return
        self._closing = True
        self.page.cleanup()
        super().done(QDialog.Accepted)
