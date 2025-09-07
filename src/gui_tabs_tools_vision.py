# gui_tabs_tools_vision.py
import sys
import textwrap
import subprocess
from pathlib import Path
import logging
import yaml
import tempfile
import os
import traceback
import gc
import time
from PIL import Image
import torch
from PySide6.QtCore import QThread, Signal as pyqtSignal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QMessageBox,
    QFileDialog, QProgressDialog, QDialog, QCheckBox
)

import module_process_images
from module_process_images import choose_image_loader
from constants import VISION_MODELS

CONFIG_FILE = 'config.yaml'


# ---------------------- config helper ----------------------
def _load_cfg() -> dict:
    p = Path(CONFIG_FILE)
    if not p.exists():
        return {}
    try:
        with p.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ---------------------- selection dialog for multi-model test ----------------------
class ModelSelectionDialog(QDialog):
    def __init__(self, models, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Vision Models")
        layout = QVBoxLayout()

        self.checkboxes = {}
        for model_name, info in models.items():
            vram_text = info.get('vram', '—')
            checkbox = QCheckBox(f"{model_name} (VRAM: {vram_text})")
            checkbox.setChecked(True)
            self.checkboxes[model_name] = checkbox
            layout.addWidget(checkbox)

        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def get_selected_models(self):
        return [model for model, checkbox in self.checkboxes.items() if checkbox.isChecked()]


# ---------------------- single-model (config-driven) processing ----------------------
class ImageProcessorThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def run(self):
        try:
            cfg = _load_cfg()
            chosen_model = ((cfg.get('vision') or {}).get('chosen_model')
                            or next(iter(VISION_MODELS.keys())))
            print(f"[Tools] Using chosen_model from config: {chosen_model}")

            # Prefer passing the model selection directly if the loader chooser supports it
            documents = None
            try:
                documents = choose_image_loader({"vision": {"chosen_model": chosen_model}})
            except TypeError:
                # Fallback for older signatures: set a module-level override if present
                try:
                    module_process_images.DEFAULT_VISION_MODEL_OVERRIDE = chosen_model
                    print("[Tools] Set module_process_images.DEFAULT_VISION_MODEL_OVERRIDE")
                except Exception:
                    pass
                documents = choose_image_loader()

            self.finished.emit(documents)
        except Exception as e:
            error_msg = f"Error in image processing: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


# ---------------------- multi-model (explicit list) processing ----------------------
class MultiModelProcessorThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

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

                    try:
                        print(f"\nProcessing with {model_name}...")
                        model_config = {"vision": {"chosen_model": model_name}}

                        loader_name = VISION_MODELS[model_name]['loader']
                        loader_class = getattr(module_process_images, loader_name)
                        loader = loader_class(model_config)

                        loader.model, loader.tokenizer, loader.processor = loader.initialize_model_and_tokenizer()
                        start_time = time.time()
                        description = loader.process_single_image(raw_image)
                        process_time = time.time() - start_time
                        description = textwrap.fill(description, width=10)
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

                    except Exception as e:
                        error_msg = f"Error processing with {model_name}: {str(e)}\n{traceback.format_exc()}"
                        results.append((model_name, error_msg, 0.0))
                        print(error_msg)
                        torch.cuda.empty_cache()
                        gc.collect()

            torch.cuda.empty_cache()
            gc.collect()
            self.finished.emit(results)
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            self.error.emit(str(e))


# ---------------------- tools tab UI ----------------------
class VisionToolSettingsTab(QWidget):
    def __init__(self):
        super().__init__()

        mainVLayout = QVBoxLayout()
        self.setLayout(mainVLayout)

        hBoxLayout = QHBoxLayout()
        mainVLayout.addLayout(hBoxLayout)

        processButton = QPushButton("Multiple Files + One Vision Model")
        hBoxLayout.addWidget(processButton)
        processButton.clicked.connect(self.confirmationBeforeProcessing)

        newButton = QPushButton("Single Image + All Vision Models")
        hBoxLayout.addWidget(newButton)
        newButton.clicked.connect(self.selectSingleImage)

        self.thread = None
        self.progress = None

    # -------- single-model path (reads Settings selection) --------
    def confirmationBeforeProcessing(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(
            "1. Create Database Tab:\n"
            "Select files you theoretically want in the vector database.\n\n"
            "2. Settings Tab:\n"
            "Select the vision model you want to test.\n\n"
            "3. Click the 'Process' button.\n\n"
            "This will test the selected vision model before actually entering the images into the vector database.\n\n"
            "Do you want to proceed?"
        )
        msgBox.setWindowTitle("Confirm Processing")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        returnValue = msgBox.exec()
        if returnValue == QMessageBox.Ok:
            self.startProcessing()

    def startProcessing(self):
        if self.thread is None:
            self.thread = ImageProcessorThread()
            self.thread.finished.connect(self.onProcessingFinished)
            self.thread.error.connect(self.onProcessingError)
            self.thread.start()

    def onProcessingFinished(self, documents):
        self.thread = None
        print(f"Processed {len(documents)} documents")
        contents = self.extract_page_content(documents)
        self.save_page_contents(contents)

    def onProcessingError(self, error_msg):
        self.thread = None
        logging.error(f"Processing error: {error_msg}")
        QMessageBox.critical(self, "Processing Error", f"An error occurred during image processing:\n\n{error_msg}")

    # -------- multi-model path (explicit dialog selection) --------
    def selectSingleImage(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.gif *.bmp *.tif *.tiff)"
        )
        if file_path:
            dialog = ModelSelectionDialog(VISION_MODELS, self)
            if dialog.exec():
                selected_models = dialog.get_selected_models()
                if not selected_models:
                    QMessageBox.warning(self, "Warning", "Please select at least one model.")
                    return

                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Information)
                msgBox.setText(
                    "Process this image with the selected vision models?\n\n"
                    "This will test each model sequentially and may take several minutes.\n"
                    "Models will be loaded and unloaded to manage memory usage.\n\n"
                    "Do you want to proceed?"
                )
                msgBox.setWindowTitle("Confirm Processing")
                msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                returnValue = msgBox.exec()

                if returnValue == QMessageBox.Ok:
                    self.progress = QProgressDialog("Processing image with multiple models...", "Cancel", 0, len(selected_models), self)
                    self.progress.setWindowModality(Qt.WindowModal)
                    self.progress.setWindowTitle("Processing")
                    self.progress.canceled.connect(self.cancelProcessing)

                    self.thread = MultiModelProcessorThread(file_path, selected_models)
                    self.thread.finished.connect(self.onMultiModelProcessingFinished)
                    self.thread.error.connect(self.onMultiModelProcessingError)
                    self.thread.progress.connect(self.progress.setValue)
                    self.thread.start()

    def cancelProcessing(self):
        if self.thread is not None and hasattr(self.thread, "cancel"):
            self.thread.cancel()

    # ---------------------- common helpers ----------------------
    def onMultiModelProcessingFinished(self, results):
        if self.progress:
            self.progress.close()
        try:
            output_file = self.save_comparison_results(self.thread.image_path, results)
            self.open_file(output_file)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving results:\n\n{str(e)}")
        self.thread = None

    def onMultiModelProcessingError(self, error_msg):
        if self.progress:
            self.progress.close()
        QMessageBox.critical(self, "Error", f"An error occurred during processing:\n\n{error_msg}")
        self.thread = None

    def extract_page_content(self, documents):
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

    def save_page_contents(self, contents):
        contents, avg_length = contents

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', encoding='utf-8', delete=False) as temp_file:
            temp_file.write(f"Average Summary Length: {avg_length:.2f} characters\n")
            temp_file.write("="*50 + "\n\n")

            for filepath, content, length in contents:
                temp_file.write(f"File Path: {filepath}\n")
                temp_file.write(f"Summary Length: {length} characters\n")
                temp_file.write("-"*50 + "\n")
                temp_file.write(f"{content}\n\n")
                temp_file.write("="*50 + "\n\n")

            temp_name = temp_file.name

        self.open_file(temp_name)

    def save_comparison_results(self, image_path, results):
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

            temp_name = temp_file.name

        self.open_file(temp_name)
        return temp_name

    def open_file(self, file_path):
        try:
            os.startfile(file_path)
        except Exception as e:
            error_msg = f"Error opening file: {e}"
            logging.error(error_msg)
            QMessageBox.warning(self, "Error", error_msg)
