from pathlib import Path

from gui.tabs_settings import settings_state
from gui.web_common.web_tab import WebTab

WEB_PAGE = Path(__file__).resolve().parent / "web" / "settings_tab.html"


class GuiSettingsTab(WebTab):
    def __init__(self, parent=None):
        settings_state.normalize_config()
        super().__init__(WEB_PAGE, parent)

    def build_state(self):
        return settings_state.build_state()

    def js_apply(self, key, value):
        result = settings_state.apply_change(key, value)
        if "state" in result:
            self.remember_state(result["state"])
        return result
