from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QTabWidget

from gui.download_model import model_downloaded_signal
from gui.web_common.web_tab import WebTab
from gui.tabs_databases import create_config
from gui.tabs_databases.create_build import BuildController, transcription_running
from gui.tabs_databases.create_files import KINDS, StagedFiles

WEB_PAGE = Path(__file__).resolve().parent / "web" / "create_tab.html"


class CreateDatabaseTab(WebTab):
    def __init__(self, parent=None):
        super().__init__(WEB_PAGE, parent)
        self.files = StagedFiles(self)
        self.build = BuildController(self, self.files)
        self.check_pdfs = True
        self.transcribing = False
        self._config = create_config.load_config()
        self._models = create_config.available_models()
        self.files.changed.connect(self.schedule_push)
        self.build.changed.connect(self.schedule_push)
        model_downloaded_signal.downloaded.connect(self._on_model_downloaded)
        self._poll = QTimer(self)
        self._poll.setInterval(1000)
        self._poll.timeout.connect(self._tick)

    def refresh(self):
        self._config = create_config.load_config()
        self._models = create_config.available_models()
        self.transcribing = transcription_running()
        self.files.refresh()

    def showEvent(self, event):
        self.refresh()
        self._poll.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._poll.stop()
        super().hideEvent(event)

    def _tick(self):
        self.files.refresh()
        transcribing = transcription_running()
        if transcribing != self.transcribing:
            self.transcribing = transcribing
            self.schedule_push()

    def _on_model_downloaded(self, model_name, model_type):
        if model_type == "vector":
            self._models = create_config.available_models()
            self.schedule_push()

    def build_state(self):
        cfg = self._config
        selected = cfg.get("EMBEDDING_MODEL_NAME")
        return {
            "files": self.files.state(),
            "kinds": [{"key": key, "label": label} for key, label, _ in KINDS] + [{"key": "other", "label": "Other"}],
            "models": self._models,
            "model": selected if any(m["path"] == selected for m in self._models) else None,
            "settings": create_config.settings_summary(cfg),
            "existing": create_config.existing_databases(),
            "name_limit": create_config.name_limit(),
            "check_pdfs": self.check_pdfs,
            "transcribing": self.transcribing,
            "build": self.build.state(),
        }

    def database_build_running(self):
        return self.build.running()

    def busy_message(self):
        return self.build.busy_message()

    def cleanup(self):
        self.files.cleanup()
        self.build.cleanup()

    def js_files(self):
        return {"version": self.files.version, "rows": self.files.entries()}

    def js_add_files(self):
        if self.build.busy():
            return {"error": "Files can't be added while a database is being created."}
        self.files.add_files(self)

    def js_add_folder(self):
        if self.build.busy():
            return {"error": "Files can't be added while a database is being created."}
        self.files.add_folder(self)

    def js_cancel_staging(self):
        self.files.cancel_staging()

    def js_remove(self, names):
        if self.build.busy():
            return {"error": "Files can't be removed while a database is being created."}
        return {"removed": self.files.remove(list(names or []))}

    def js_open(self, name):
        self.files.open(name)

    def js_dismiss_notice(self):
        self.files.dismiss_notice()

    def js_select_model(self, path):
        if self.build.busy():
            return {"error": "The model can't be changed while a database is being created."}
        if path and not any(m["path"] == path for m in self._models):
            return {"error": "That model is no longer available."}
        create_config.select_model(path or None)
        self._config = create_config.load_config()
        self.schedule_push()
        return {"ok": True}

    def js_set_check_pdfs(self, value):
        self.check_pdfs = bool(value)
        self.schedule_push()

    def js_create(self, name):
        self.refresh()
        selected = self._config.get("EMBEDDING_MODEL_NAME")
        model = next((m for m in self._models if m["path"] == selected), None)
        result = self.build.start(self, name, model["folder"] if model else Path(selected or "").name,
                                  self.check_pdfs, self._config, model["name"] if model else None)
        self._config = create_config.load_config()
        self.schedule_push()
        return result

    def js_cancel_build(self):
        self.build.cancel()

    def js_dismiss_result(self):
        self.build.dismiss()

    def js_log(self):
        return {"lines": self.build.log()}

    def js_open_report(self):
        self.build.open_report()

    def js_remove_ocr_files(self):
        self.build.remove_ocr_files()

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
