import gc
import logging
import os
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

import torch
import yaml
from PIL import Image
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QFileDialog

from core.constants import PROJECT_ROOT, VISION_MODELS
from core.utilities import fallback_if_unavailable, runs_on_this_hardware

CONFIG_FILE = 'config.yaml'
ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']


def process_images_module():
    import modules.process_images as module_process_images

    logging.getLogger("transformers").setLevel(logging.ERROR)
    return module_process_images


def _load_cfg() -> dict:
    p = Path(CONFIG_FILE)
    if not p.exists():
        return {}
    try:
        with p.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def chosen_vision_model():
    cfg = _load_cfg()
    return fallback_if_unavailable((cfg.get('vision') or {}).get('chosen_model'),
                                   VISION_MODELS, next(iter(VISION_MODELS.keys())))


def count_images():
    try:
        return sum(1 for f in (PROJECT_ROOT / "Docs_for_DB").iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS)
    except OSError:
        return 0


class ImageProcessorThread(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, chosen_model, parent=None):
        super().__init__(parent)
        self.chosen_model = chosen_model

    def run(self):
        try:
            print(f"[Tools] Using chosen_model from config: {self.chosen_model}")
            documents = process_images_module().choose_image_loader({"vision": {"chosen_model": self.chosen_model}})
            self.finished.emit(documents)
        except Exception as e:
            error_msg = f"Error in image processing: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class MultiModelProcessorThread(QThread):
    finished = Signal(list)
    error = Signal(str)
    progress = Signal(int)
    model_started = Signal(int, str)
    model_completed = Signal(int, str, float)
    model_failed = Signal(int, str)

    def __init__(self, image_path, selected_models):
        super().__init__()
        self.image_path = image_path
        self.selected_models = selected_models
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            results = []
            with Image.open(self.image_path) as raw_image:
                for i, model_name in enumerate(self.selected_models):
                    if self.is_cancelled:
                        print("\nProcessing cancelled by user")
                        torch.cuda.empty_cache()
                        gc.collect()
                        break

                    self.model_started.emit(i, model_name)

                    try:
                        print(f"\nProcessing with {model_name}...")
                        model_config = {"vision": {"chosen_model": model_name}}

                        loader_name = VISION_MODELS[model_name]['loader']
                        loader_class = getattr(process_images_module(), loader_name)
                        loader = loader_class(model_config)

                        loader.model, loader.tokenizer, loader.processor = loader.initialize_model_and_tokenizer()
                        start_time = time.time()
                        description = loader.process_single_image(raw_image)
                        process_time = time.time() - start_time
                        description = textwrap.fill(description, width=100)
                        results.append((model_name, description, process_time))

                        if hasattr(loader, 'model') and loader.model is not None:
                            loader.model.cpu()
                            del loader.model
                        if hasattr(loader, 'tokenizer') and loader.tokenizer is not None:
                            del loader.tokenizer
                        if hasattr(loader, 'processor') and loader.processor is not None:
                            del loader.processor

                        torch.cuda.empty_cache()
                        gc.collect()

                        print(f"Completed {model_name}")
                        self.progress.emit(i + 1)
                        self.model_completed.emit(i, model_name, process_time)

                    except Exception as e:
                        error_msg = f"Error processing with {model_name}: {str(e)}\n{traceback.format_exc()}"
                        results.append((model_name, error_msg, 0.0))
                        print(error_msg)
                        torch.cuda.empty_cache()
                        gc.collect()
                        self.model_failed.emit(i, model_name)

            torch.cuda.empty_cache()
            gc.collect()
            self.finished.emit(results)
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            self.error.emit(str(e))


def extract_page_content(documents):
    contents = []
    total_length = 0

    for doc in documents:
        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
            content = doc.page_content
            filepath = doc.metadata.get('source', doc.metadata.get('file_path', doc.metadata.get('file_name', 'Unknown filepath')))
        elif isinstance(doc, dict):
            content = doc.get("page_content", "Document is missing 'page_content'.")
            filepath = doc.get("metadata", {}).get('source',
                     doc.get("metadata", {}).get('file_path',
                     doc.get("metadata", {}).get('file_name', 'Unknown filepath')))
        else:
            content = "Document is missing 'page_content'."
            filepath = 'Unknown filepath'

        content_length = len(content)
        total_length += content_length
        wrapped_content = textwrap.fill(content, width=100)
        contents.append((filepath, wrapped_content, content_length))

    avg_length = total_length / len(documents) if documents else 0
    return contents, avg_length


def save_page_contents(contents, avg_length):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as temp_file:
        temp_file.write(f"Average Summary Length: {avg_length:.2f} characters\n")
        temp_file.write("="*50 + "\n\n")

        for filepath, content, length in contents:
            temp_file.write(f"File Path: {filepath}\n")
            temp_file.write(f"Summary Length: {length} characters\n")
            temp_file.write("-"*50 + "\n")
            temp_file.write(f"{content}\n\n")
            temp_file.write("="*50 + "\n\n")

        return temp_file.name


def save_comparison_results(image_path, results):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as temp_file:
        model_col_width = 23
        count_col_width = 12
        time_col_width = 12
        speed_col_width = 12

        temp_file.write(f"Image Path: {image_path}\n")
        temp_file.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        chunk_advice = (
            "Remember to adjust your 'chunk size' setting to exceed the longest image summary that you expect. "
            "For large bodies of text (e.g. from a .pdf) splitting/overlapping chunks of text is fine, but for image "
            "summaries you want any/all summaries to fit within a single chunk that will be put into the vector database."
        )
        temp_file.write(textwrap.fill(chunk_advice, width=100) + "\n\n")

        temp_file.write("Model Performance Comparison Table:\n")
        temp_file.write("+" + "-"*model_col_width + "+" + "-"*count_col_width + "+" +
                       "-"*time_col_width + "+" + "-"*speed_col_width + "+\n")
        temp_file.write("|" + "Model Name".center(model_col_width) + "|" +
                       "Char Count".center(count_col_width) + "|" +
                       "Time (sec)".center(time_col_width) + "|" +
                       "Char/Sec".center(speed_col_width) + "|\n")
        temp_file.write("+" + "-"*model_col_width + "+" + "-"*count_col_width + "+" +
                       "-"*time_col_width + "+" + "-"*speed_col_width + "+\n")

        for model_name, description, process_time in results:
            char_count = len(description)
            chars_per_sec = char_count / process_time if process_time > 0 else 0

            temp_file.write("|" + model_name.ljust(model_col_width) + "|" +
                          str(char_count).center(count_col_width) + "|" +
                          f"{process_time:.2f}".center(time_col_width) + "|" +
                          f"{chars_per_sec:.1f}".center(speed_col_width) + "|\n")

        temp_file.write("+" + "-"*model_col_width + "+" + "-"*count_col_width + "+" +
                       "-"*time_col_width + "+" + "-"*speed_col_width + "+\n\n")

        for model_name, description, process_time in results:
            char_count = len(description)
            chars_per_sec = char_count / process_time if process_time > 0 else 0

            temp_file.write(f"Model: {model_name}\n")
            temp_file.write(f"Summary Length: {char_count}\n")
            temp_file.write(f"Processing Time: {process_time:.2f} seconds\n")
            temp_file.write(f"Characters per Second: {chars_per_sec:.1f}\n")
            temp_file.write("="*50 + "\n")
            if description.strip():
                temp_file.write(textwrap.fill(description, width=100) + "\n\n")
            else:
                temp_file.write("[No output generated]\n\n")
            temp_file.write("-"*50 + "\n\n")

        return temp_file.name


def open_file(file_path):
    try:
        os.startfile(file_path)
        return None
    except Exception as e:
        error_msg = f"Error opening file: {e}"
        logging.error(error_msg)
        return error_msg


class VisionTool(QObject):
    changed = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.models = [
            {"name": name, "vram": info.get('vram', '—'), "available": runs_on_this_hardware(info)}
            for name, info in VISION_MODELS.items()
        ]
        self.selected = [m["name"] for m in self.models if m["available"]]
        self.chosen = None
        self.images = 0
        self.file = None
        self.task = None
        self.worker = None
        self.started = None
        self.progress = []
        self.cancelling = False
        self.result = None
        self.refresh()

    def refresh(self):
        self.chosen = chosen_vision_model()
        self.images = count_images()

    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def state(self):
        file_state = None
        if self.file:
            path = Path(self.file)
            file_state = {"name": path.name, "path": str(path.absolute())}
        return {
            "chosen": self.chosen,
            "images": self.images,
            "models": self.models,
            "selected": [name for name in self.selected],
            "file": file_state,
            "task": self.task if self.running() else None,
            "started": self.started,
            "progress": self.progress,
            "cancelling": self.cancelling,
            "result": self.result,
        }

    def set_selected(self, names):
        if self.running():
            return
        available = {m["name"] for m in self.models if m["available"]}
        wanted = set(names or [])
        self.selected = [m["name"] for m in self.models if m["name"] in wanted and m["name"] in available]
        self.changed.emit()

    def choose_image(self, parent_widget):
        if self.running():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            parent_widget,
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.gif *.bmp *.tif *.tiff)"
        )
        if file_path:
            self.file = file_path
            if self.result and self.result.get("kind") == "compare":
                self.result = None
            self.changed.emit()

    def make_summary_worker(self, chosen_model):
        return ImageProcessorThread(chosen_model)

    def make_compare_worker(self, image_path, selected_models):
        return MultiModelProcessorThread(image_path, selected_models)

    def _busy(self):
        if self.running():
            self.result = {"kind": self.task, "ok": False,
                           "message": "An image-processing task is already running. Please wait for it to finish before starting another."}
            self.changed.emit()
            return True
        return False

    def summarize(self):
        if self._busy():
            return
        self.refresh()
        if not self.images:
            self.result = {"kind": "summarize", "ok": False,
                           "message": "Add images on the Create Database tab first, then run the test again."}
            self.changed.emit()
            return
        self.task = "summarize"
        self.result = None
        self.progress = []
        self.cancelling = False
        self.started = time.time()
        self.worker = self.make_summary_worker(self.chosen)
        self.worker.finished.connect(self._on_summary_finished)
        self.worker.error.connect(self._on_summary_error)
        self.worker.start()
        self.changed.emit()

    def _release_worker(self):
        worker = self.worker
        if worker is not None:
            worker.wait()
            self.worker = None

    def _on_summary_finished(self, documents):
        self._release_worker()
        elapsed = time.time() - (self.started or time.time())
        print(f"Processed {len(documents)} documents")
        if not documents:
            self.result = {"kind": "summarize", "ok": False,
                           "message": "No summaries were produced. The command prompt window shows what went wrong."}
            self.changed.emit()
            return
        contents, avg_length = extract_page_content(documents)
        path = save_page_contents(contents, avg_length)
        error = open_file(path)
        self.result = {
            "kind": "summarize", "ok": True, "model": self.chosen, "count": len(documents),
            "average": round(avg_length), "longest": max(length for _, _, length in contents),
            "seconds": round(elapsed, 1), "path": path, "open_error": error,
        }
        self.changed.emit()

    def _on_summary_error(self, error_msg):
        self._release_worker()
        logging.error(f"Processing error: {error_msg}")
        self.result = {"kind": "summarize", "ok": False,
                       "message": f"An error occurred during image processing: {error_msg.splitlines()[0] if error_msg else ''}"}
        self.changed.emit()

    def compare(self):
        if self._busy():
            return
        if not self.file:
            self.result = {"kind": "compare", "ok": False, "message": "Choose an image first."}
            self.changed.emit()
            return
        if not self.selected:
            self.result = {"kind": "compare", "ok": False, "message": "Please select at least one model."}
            self.changed.emit()
            return
        names = list(self.selected)
        self.task = "compare"
        self.result = None
        self.cancelling = False
        self.started = time.time()
        self.progress = [{"name": name, "status": "pending", "seconds": None} for name in names]
        self.worker = self.make_compare_worker(self.file, names)
        self.worker.finished.connect(self._on_compare_finished)
        self.worker.error.connect(self._on_compare_error)
        self.worker.model_started.connect(self._on_model_started)
        self.worker.model_completed.connect(self._on_model_completed)
        self.worker.model_failed.connect(self._on_model_failed)
        self.worker.start()
        self.changed.emit()

    def _set_progress(self, index, status, seconds=None):
        if 0 <= index < len(self.progress):
            self.progress[index] = {"name": self.progress[index]["name"], "status": status,
                                    "seconds": None if seconds is None else round(seconds, 1),
                                    "since": time.time() if status == "running" else None}
            self.changed.emit()

    def _on_model_started(self, index, name):
        self._set_progress(index, "running")

    def _on_model_completed(self, index, name, elapsed):
        self._set_progress(index, "done", elapsed)

    def _on_model_failed(self, index, name):
        self._set_progress(index, "failed")

    def cancel(self):
        if self.running() and self.task == "compare":
            self.cancelling = True
            self.worker.cancel()
            self.changed.emit()

    def _on_compare_finished(self, results):
        image_path = self.worker.image_path if self.worker is not None else self.file
        self._release_worker()
        cancelled = self.cancelling
        self.cancelling = False
        self.progress = [p if p["status"] != "pending" else {**p, "status": "skipped"} for p in self.progress]
        if not results:
            self.result = {"kind": "compare", "ok": False, "message": "Cancelled before any model finished."}
            self.changed.emit()
            return
        try:
            path = save_comparison_results(image_path, results)
        except Exception as e:
            self.result = {"kind": "compare", "ok": False, "message": f"An error occurred while saving results: {e}"}
            self.changed.emit()
            return
        error = open_file(path)
        rows = []
        for model_name, description, process_time in results:
            failed = process_time == 0.0 and description.startswith("Error processing with")
            chars = 0 if failed else len(description)
            rows.append({"name": model_name, "chars": chars, "seconds": round(process_time, 1),
                         "speed": round(chars / process_time, 1) if process_time > 0 else 0, "failed": failed})
        self.result = {"kind": "compare", "ok": True, "cancelled": cancelled, "rows": rows, "path": path, "open_error": error}
        self.changed.emit()

    def _on_compare_error(self, error_msg):
        self._release_worker()
        self.cancelling = False
        self.result = {"kind": "compare", "ok": False, "message": f"An error occurred during processing: {error_msg}"}
        self.changed.emit()

    def open_result(self):
        if self.result and self.result.get("path") and Path(self.result["path"]).exists():
            error = open_file(self.result["path"])
            if error:
                self.result = {**self.result, "open_error": error}
                self.changed.emit()

    def busy_message(self):
        return None

    def cleanup(self):
        if self.running():
            if self.task == "compare":
                self.worker.cancel()
            self.worker.wait(5000)
