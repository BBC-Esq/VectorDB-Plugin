import time
from pathlib import Path
import fitz
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, 
    QComboBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import QThread, Signal
from module_ocr import process_documents

def get_pdf_page_count(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return 0

def run_ocr_process(pdf_path, backend):
    try:
        process_documents(
            pdf_paths=Path(pdf_path),
            backend=backend,
        )
        return True, None
    except Exception as e:
        return False, str(e)

class OcrWorkerThread(QThread):
    finished_signal = Signal(bool, str, float)

    def __init__(self, pdf_path, backend, parent=None):
        super().__init__(parent)
        self.pdf_path = pdf_path
        self.backend = backend

    def run(self):
        start_time = time.time()
        result = run_ocr_process(self.pdf_path, self.backend)
        elapsed_time = time.time() - start_time
        self.finished_signal.emit(*result, elapsed_time)

class OCRToolSettingsTab(QWidget):
    # Simplified to only include Tesseract
    ENGINE_MAPPING = {
        "Tesseract": "tesseract"
    }

    def __init__(self):
        super().__init__()
        self.selected_pdf_file = None
        self.create_layout()
        self.setButtons(True)
        self.worker_thread = None

    def create_layout(self):
        main_layout = QVBoxLayout()

        engine_selection_hbox = QHBoxLayout()

        engine_label = QLabel("OCR Engine")
        engine_selection_hbox.addWidget(engine_label)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["Tesseract"])
        self.engine_combo.setCurrentText("Tesseract")
        engine_selection_hbox.addWidget(self.engine_combo)

        self.select_pdf_button = QPushButton("Choose PDF")
        self.select_pdf_button.clicked.connect(self.select_pdf_file)
        engine_selection_hbox.addWidget(self.select_pdf_button)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.start_ocr_process)
        engine_selection_hbox.addWidget(self.process_button)

        engine_selection_hbox.setStretchFactor(engine_label, 1)
        engine_selection_hbox.setStretchFactor(self.engine_combo, 2)
        engine_selection_hbox.setStretchFactor(self.select_pdf_button, 1)
        engine_selection_hbox.setStretchFactor(self.process_button, 1)

        main_layout.addLayout(engine_selection_hbox)

        self.file_path_label = QLabel("No PDF file selected")
        main_layout.addWidget(self.file_path_label)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

    def setButtons(self, enabled):
        self.select_pdf_button.setEnabled(enabled)
        self.process_button.setEnabled(enabled)
        self.engine_combo.setEnabled(enabled)
        if enabled:
            self.status_label.setText("")

    def select_pdf_file(self):
        current_dir = Path.cwd()
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Select PDF File", 
            str(current_dir),
            "PDF Files (*.pdf)"
        )
        if file_name:
            file_path = Path(file_name)
            short_path = f"...{file_path.parent.name}/{file_path.name}"
            self.file_path_label.setText(short_path)
            self.file_path_label.setToolTip(str(file_path.absolute()))
            self.selected_pdf_file = file_name
            self.status_label.setText("")

    def show_error_message(self, message):
        self.status_label.setStyleSheet("color: red;")
        self.status_label.setText("Error: OCR process failed")
        QMessageBox.critical(self, "Error", f"OCR process failed:\n{message}")

    def show_success_message(self):
        self.status_label.setStyleSheet("color: #4CAF50;")

        minutes, seconds = divmod(self.elapsed_time, 60)
        time_str = f"{int(minutes)}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"
        self.status_label.setText(f"Success! Completed in {time_str}")

        if not self.selected_pdf_file:
            return

        original_file = Path(self.selected_pdf_file)
        processed_file = original_file.with_stem(f"{original_file.stem}_OCR").with_suffix(".pdf")

        if processed_file.exists():
            file_link = f'<a href="file:///{processed_file}" style="color: #4CAF50; text-decoration: none;">Open New File</a>'
        else:
            file_link = "The processed file could not be found."

        QMessageBox.information(
            self,
            "Success!",
            f"""Processing completed in {time_str}!<br><br>
            A new <b>.pdf</b> ending in <b>'_OCR'</b> has been saved
            in the same directory as the original file.<br><br>

            {file_link}
            """
        )

    def start_ocr_process(self):
        if not self.selected_pdf_file:
            QMessageBox.warning(self, "Warning", "Please select a PDF file first.")
            return

        selected_engine = self.engine_combo.currentText()
        backend = self.ENGINE_MAPPING[selected_engine]

        self.status_label.setStyleSheet("color: #0074D9;")
        self.status_label.setText(f"Processing with {selected_engine}...")
        print(f"Starting OCR process for {self.selected_pdf_file}")

        self.setButtons(False)

        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.wait()

        self.worker_thread = OcrWorkerThread(self.selected_pdf_file, backend)
        self.worker_thread.finished_signal.connect(self.ocr_finished)
        self.worker_thread.start()

    def ocr_finished(self, success, message, elapsed_time):
        self.setButtons(True)

        self.elapsed_time = elapsed_time

        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None

        from PySide6.QtCore import QTimer
        QTimer.singleShot(1000, lambda: self._show_completion_message(success, message))

    def _show_completion_message(self, success, message):
        if success:
            self.show_success_message()
        else:
            self.show_error_message(message)