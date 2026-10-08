import re
import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication, QFileDialog

from core.constants import PROJECT_ROOT, WHISPER_MODELS
from core.utilities import cuda_usable, has_bfloat16_support, my_cprint

BUILD_RUNNING_MESSAGE = (
    "A vector database is being created. Transcribe after it finishes, because a "
    "transcript saved during a build would be removed when the build ends."
)

PRECISIONS = ["float32", "bfloat16", "float16"]


class TranscriptionWorkerThread(QThread):
    finished_signal = Signal(bool, str)

    def __init__(self, model_key, batch_size, audio_file, parent=None):
        super().__init__(parent)
        self.model_key = model_key
        self.batch_size = batch_size
        self.audio_file = audio_file

    def run(self):
        try:
            from modules.transcribe import WhisperTranscriber

            transcriber = WhisperTranscriber(
                model_key=self.model_key,
                batch_size=self.batch_size
            )
            transcriber.start_transcription_process(self.audio_file)
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


def model_table():
    cuda_available = cuda_usable()
    bfloat16_supported = cuda_available and has_bfloat16_support()
    table = {}
    for model_key, model_info in WHISPER_MODELS.items():
        precision = model_info['precision']
        available = (precision == 'float32'
                     or (precision == 'bfloat16' and bfloat16_supported)
                     or (precision == 'float16' and cuda_available))
        table.setdefault(model_info['name'], {})[precision] = {"key": model_key, "available": available}
    return table


def database_build_running():
    return any(callable(getattr(w, "database_build_running", None)) and w.database_build_running()
               for w in QApplication.allWidgets())


class TranscribeTool(QObject):
    changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.table = model_table()
        self.model = next(iter(self.table), None)
        self.precision = self._first_precision(self.model)
        self.batch = 8
        self.file = None
        self.worker = None
        self.started = None
        self.result = None

    def _first_precision(self, model):
        for precision in PRECISIONS:
            entry = self.table.get(model, {}).get(precision)
            if entry and entry["available"]:
                return precision
        return None

    def model_key(self):
        entry = self.table.get(self.model, {}).get(self.precision)
        return entry["key"] if entry else None

    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def state(self):
        file_state = None
        if self.file:
            path = Path(self.file)
            file_state = {"name": path.name, "path": str(path.absolute())}
        precisions = self.table.get(self.model, {})
        return {
            "models": list(self.table),
            "model": self.model,
            "precisions": [
                {"value": p, "available": bool(precisions.get(p, {}).get("available"))}
                for p in PRECISIONS if p in precisions
            ],
            "precision": self.precision,
            "batch": self.batch,
            "file": file_state,
            "running": self.running(),
            "started": self.started,
            "result": self.result,
        }

    def set_model(self, value):
        if value in self.table and not self.running():
            self.model = value
            entry = self.table[value].get(self.precision)
            if not (entry and entry["available"]):
                self.precision = self._first_precision(value)
            self.changed.emit()

    def set_precision(self, value):
        entry = self.table.get(self.model, {}).get(value)
        if entry and entry["available"] and not self.running():
            self.precision = value
            self.changed.emit()

    def set_batch(self, value):
        try:
            number = int(str(value).strip())
        except ValueError:
            return "Batch size must be a whole number from 1 to 150."
        if not 1 <= number <= 150:
            return "Batch size must be a whole number from 1 to 150."
        self.batch = number
        self.changed.emit()
        return None

    def choose_file(self, parent_widget):
        if self.running():
            return
        file_name, _ = QFileDialog.getOpenFileName(parent_widget, "Select Audio File", str(Path.cwd()))
        if file_name:
            self.file = file_name
            self.result = None
            self.changed.emit()

    def make_worker(self, model_key, batch_size, audio_file):
        return TranscriptionWorkerThread(model_key, batch_size, audio_file)

    def start(self):
        if self.running():
            return
        if not self.file:
            self.result = {"ok": False, "message": "Choose an audio file first."}
            self.changed.emit()
            return
        if database_build_running():
            self.result = {"ok": False, "message": BUILD_RUNNING_MESSAGE}
            self.changed.emit()
            return
        self.result = None
        self.started = time.time()
        self.worker = self.make_worker(self.model_key(), self.batch, self.file)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()
        self.changed.emit()

    def _transcript_name(self):
        pattern = re.compile(rf"{re.escape(Path(self.file).stem)}( \(\d+\))?\.json")
        try:
            candidates = [p for p in (PROJECT_ROOT / "Docs_for_DB").iterdir()
                          if pattern.fullmatch(p.name) and p.stat().st_mtime >= (self.started or 0) - 1]
        except OSError:
            return None
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime).name

    def _on_finished(self, success, message):
        worker = self.worker
        if worker is not None:
            worker.quit()
            worker.wait()
            self.worker = None
        if success:
            my_cprint("Transcription created and ready to be input into vector database.", 'green')
            name = self._transcript_name()
            saved = f"The transcript was saved as {name} in the Docs_for_DB folder" if name else "The transcript was saved in the Docs_for_DB folder"
            self.result = {"ok": True, "message": f"{saved}, so it will be included in the next database you create."}
        else:
            my_cprint(f"Transcription failed: {message}", 'red')
            self.result = {"ok": False, "message": f"Transcription failed: {message}"}
        self.changed.emit()
