import re
from functools import lru_cache
from pathlib import Path

import torch

from core.utilities import cuda_usable
from gui.tabs_tools.backups import BackupTool
from gui.tabs_tools.gpus import VRAM_SIZES, gpu_rows
from gui.tabs_tools.ocr import OcrTool
from gui.tabs_tools.scrape import ScrapeTool
from gui.tabs_tools.transcribe import TranscribeTool
from gui.tabs_tools.vision import VisionTool
from gui.web_common.web_tab import WebTab

WEB_PAGE = Path(__file__).resolve().parent / "web" / "tools_tab.html"


@lru_cache(maxsize=1)
def local_gpu():
    if not cuda_usable():
        return None
    return re.sub(r"^NVIDIA\s+", "", torch.cuda.get_device_name(0)).strip()


class ToolsTab(WebTab):
    def __init__(self, parent=None):
        super().__init__(WEB_PAGE, parent)
        self.transcribe = TranscribeTool(self)
        self.scrape = ScrapeTool(self)
        self.vision = VisionTool(self)
        self.ocr = OcrTool(self)
        self.backups = BackupTool(self)
        self.tools = {
            "transcribe": self.transcribe,
            "scrape": self.scrape,
            "vision": self.vision,
            "ocr": self.ocr,
            "backups": self.backups,
        }
        for tool in self.tools.values():
            tool.changed.connect(self.schedule_push)

    def build_state(self):
        state = {name: tool.state() for name, tool in self.tools.items()}
        state["gpu"] = {"local": local_gpu(), "sizes": VRAM_SIZES}
        return state

    def showEvent(self, event):
        self.scrape.refresh()
        self.vision.refresh()
        self.backups.refresh()
        super().showEvent(event)

    def transcription_running(self):
        return self.transcribe.running()

    def busy_message(self):
        for tool in self.tools.values():
            message = tool.busy_message() if hasattr(tool, "busy_message") else None
            if message:
                return message
        return None

    def cleanup(self):
        for tool in self.tools.values():
            if hasattr(tool, "cleanup"):
                tool.cleanup()

    def js_transcribe_model(self, value):
        self.transcribe.set_model(value)

    def js_transcribe_precision(self, value):
        self.transcribe.set_precision(value)

    def js_transcribe_batch(self, value):
        error = self.transcribe.set_batch(value)
        return {"error": error} if error else {"ok": True}

    def js_transcribe_choose(self):
        self.transcribe.choose_file(self)

    def js_transcribe_start(self):
        self.transcribe.start()

    def js_scrape_select(self, name):
        self.scrape.select(name)

    def js_scrape_start(self, name):
        self.scrape.start(self, name)

    def js_scrape_cancel(self, name):
        self.scrape.cancel(name)

    def js_scrape_open(self, name):
        self.scrape.open_folder(self, name)

    def js_scrape_dismiss(self, name):
        self.scrape.dismiss(name)

    def js_vision_models(self, names):
        self.vision.set_selected(names)

    def js_vision_choose(self):
        self.vision.choose_image(self)

    def js_vision_summarize(self):
        self.vision.summarize()

    def js_vision_compare(self):
        self.vision.compare()

    def js_vision_cancel(self):
        self.vision.cancel()

    def js_vision_open(self):
        self.vision.open_result()

    def js_ocr_engine(self, value):
        self.ocr.set_engine(value)

    def js_ocr_choose(self):
        self.ocr.choose_pdf(self)

    def js_ocr_start(self):
        self.ocr.start()

    def js_ocr_open(self):
        self.ocr.open_output()

    def js_backup(self):
        self.backups.backup(self)

    def js_restore(self):
        self.backups.restore(self)

    def js_gpus(self, min_vram, max_vram):
        if int(min_vram) > int(max_vram):
            return {"error": "Minimum V-RAM value cannot exceed maximum V-RAM value."}
        return {"rows": gpu_rows(int(min_vram), int(max_vram))}
