import os
import time
from pathlib import Path

import fitz
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog

from modules.ocr import process_documents

ENGINES = [("rapidocr", "RapidOCR"), ("tesseract", "Tesseract")]


def get_pdf_page_count(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return 0


def run_ocr_process(pdf_path, backend, on_progress=None):
    try:
        events = process_documents(pdf_paths=Path(pdf_path), backend=backend, on_progress=on_progress)
        return True, None, events if isinstance(events, dict) else {}
    except Exception as e:
        return False, str(e), {}


def _page_list(pages, limit=10):
    shown = ", ".join(str(p) for p in pages[:limit])
    return shown + ", ..." if len(pages) > limit else shown


def summarize_events(events):
    events = events or {}

    def entries(key):
        return [e for e in events.get(key, []) if isinstance(e, dict)]

    def pages(items):
        return sorted({e.get('page') for e in items if e.get('page')})

    warnings, infos = [], []
    lc = pages(entries('lowconf'))
    if lc:
        warnings.append(f"{len(lc)} low-confidence page(s): {_page_list(lc)} (worth a manual review)")
    nt = entries('notext')
    nt_sus = pages([e for e in nt if e.get('ink_frac', 0) >= 0.002])
    nt_blank = pages([e for e in nt if e.get('ink_frac', 0) < 0.002])
    if nt_sus:
        warnings.append(f"{len(nt_sus)} page(s) with visible content but no OCR text: {_page_list(nt_sus)}")
    if nt_blank:
        infos.append(f"{len(nt_blank)} blank page(s): {_page_list(nt_blank)}")
    pe = pages(entries('pageerror'))
    if pe:
        warnings.append(f"{len(pe)} page(s) failed OCR (image kept, no text layer): {_page_list(pe)}")
    mm = pages(entries('datamismatch'))
    if mm:
        warnings.append(f"{len(mm)} page(s) with inconsistent OCR data (partial text kept): {_page_list(mm)}")
    for e in entries('verifyfail'):
        warnings.append(f"verification: {e.get('msg', 'output verification warning')}")
    for e in entries('fileerror'):
        warnings.append(f"file failed: {e.get('error', 'unknown error')}")
    orp = pages(entries('oriented'))
    if orp:
        infos.append(f"{len(orp)} page(s) auto-rotated to read: {_page_list(orp)}")
    return warnings, infos


def format_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"


class OcrWorkerThread(QThread):
    finished_signal = Signal(bool, str, float, object)
    progress_signal = Signal(str, int)

    def __init__(self, pdf_path, backend, parent=None):
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.backend = backend

    def run(self):
        start_time = time.time()
        success, message, events = run_ocr_process(self.pdf_path, self.backend, self.progress_signal.emit)
        elapsed_time = time.time() - start_time
        self.finished_signal.emit(success, message or "", elapsed_time, events)


class OcrTool(QObject):
    changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.engine = "rapidocr"
        self.file = None
        self.pages = 0
        self.worker = None
        self.started = None
        self.progress = None
        self.result = None

    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def state(self):
        file_state = None
        if self.file:
            path = Path(self.file)
            file_state = {"name": path.name, "path": str(path.absolute()), "pages": self.pages}
        return {
            "engines": [{"value": value, "label": label} for value, label in ENGINES],
            "engine": self.engine,
            "file": file_state,
            "running": self.running(),
            "started": self.started,
            "progress": self.progress,
            "result": self.result,
        }

    def set_engine(self, value):
        if value in dict(ENGINES) and not self.running():
            self.engine = value
            self.changed.emit()

    def choose_pdf(self, parent_widget):
        if self.running():
            return
        file_name, _ = QFileDialog.getOpenFileName(parent_widget, "Select PDF File", str(Path.cwd()), "PDF Files (*.pdf)")
        if file_name:
            self.file = file_name
            self.pages = get_pdf_page_count(file_name)
            self.result = None
            self.changed.emit()

    def make_worker(self, pdf_path, backend):
        return OcrWorkerThread(pdf_path, backend)

    def start(self):
        if self.running():
            return
        if not self.file:
            self.result = {"ok": False, "message": "Choose a PDF file first."}
            self.changed.emit()
            return
        print(f"Starting OCR process for {self.file}")
        self.result = None
        self.started = time.time()
        self.progress = {"done": 0, "total": self.pages or 0}
        self.worker = self.make_worker(self.file, self.engine)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()
        self.changed.emit()

    def _on_progress(self, kind, value):
        if self.progress is None:
            return
        if kind == "total":
            self.progress = {"done": 0, "total": int(value)}
        elif kind == "update":
            self.progress = {"done": self.progress["done"] + int(value), "total": self.progress["total"]}
        self.changed.emit()

    def output_path(self):
        if not self.file:
            return None
        original = Path(self.file)
        return original.with_stem(f"{original.stem}_OCR").with_suffix(".pdf")

    def _on_finished(self, success, message, elapsed_time, events):
        worker = self.worker
        if worker is not None:
            worker.quit()
            worker.wait()
            self.worker = None
        self.progress = None
        if success:
            warnings, infos = summarize_events(events if isinstance(events, dict) else {})
            output = self.output_path()
            self.result = {
                "ok": True,
                "time": format_seconds(elapsed_time),
                "warnings": warnings,
                "infos": infos,
                "output": str(output) if output and output.exists() else None,
                "output_name": output.name if output else "",
            }
        else:
            self.result = {"ok": False, "message": f"OCR failed: {message}"}
        self.changed.emit()

    def open_output(self):
        if self.result and self.result.get("output") and Path(self.result["output"]).exists():
            os.startfile(self.result["output"])
