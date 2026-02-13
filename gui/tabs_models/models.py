import threading
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
   QWidget, QLabel, QGridLayout, QVBoxLayout, QGroupBox, QPushButton, QRadioButton, QButtonGroup
)

from core.constants import VECTOR_MODELS, TOOLTIPS
from gui.download_model import ModelDownloader, model_downloaded_signal

class VectorModelsTab(QWidget):
    def __init__(self, parent=None):
       super().__init__(parent)
       self.main_layout = QVBoxLayout()
       self.setLayout(self.main_layout)

       self.group_boxes = {}
       self.downloaded_labels = {}
       self.model_radiobuttons = QButtonGroup(self)
       self.model_radiobuttons.setExclusive(True)
       self.stretch_factors = {
           'BAAI': 4,
           'intfloat': 4,
           'IBM': 3,
           'infly': 3,
           'Snowflake': 3,
           'Qwen': 4,
           'Google': 2,
       }

       models_dir = Path('Models')
       if not models_dir.exists():
           models_dir.mkdir(parents=True)

       vector_models_dir = models_dir / "vector"
       if not vector_models_dir.exists():
           vector_models_dir.mkdir(parents=True)

       existing_vector_directories = {d.name for d in vector_models_dir.iterdir() if d.is_dir()}

       headers = ["Select", "Model Name", "Original Precision", "Parameters", "Dimensions", "Max Sequence", "Size (MB)", "Downloaded"]
       column_stretch_factors = [1, 2, 2, 1, 1, 1, 1, 1]

       def add_centered_widget(grid, widget, row, col):
           grid.addWidget(widget, row, col, alignment=Qt.AlignCenter)

       row_counter = 1
       for vendor, models in VECTOR_MODELS.items():
           group_box = QGroupBox(vendor)

           group_box.setStyleSheet("""
               QGroupBox::title {
                   subcontrol-origin: margin;
                   padding: 0 5px;
                   font-weight: bold;
                   color: #00bf9e;
               }
           """)

           group_layout = QGridLayout()
           group_layout.setVerticalSpacing(0)
           group_layout.setHorizontalSpacing(0)
           group_box.setLayout(group_layout)
           group_layout.setContentsMargins(0, 10, 0, 0)
           
           size_policy = group_box.sizePolicy()
           size_policy.setVerticalStretch(self.stretch_factors.get(vendor, 1))
           group_box.setSizePolicy(size_policy)
           
           self.group_boxes[vendor] = group_box

           for col, header in enumerate(headers):
               header_label = QLabel(header)
               header_label.setAlignment(Qt.AlignCenter)
               header_label.setStyleSheet("text-decoration: underline;")
               header_label.setToolTip(TOOLTIPS.get(f"VECTOR_MODEL_{header.upper().replace(' ', '_')}", ""))
               group_layout.addWidget(header_label, 0, col)

           for col, stretch_factor in enumerate(column_stretch_factors):
               group_layout.setColumnStretch(col, stretch_factor)

           for model in models:
               model_info = model
               grid = group_box.layout()
               row = grid.rowCount()

               radiobutton = QRadioButton()
               radiobutton.setToolTip(TOOLTIPS.get("VECTOR_MODEL_SELECT", ""))
               self.model_radiobuttons.addButton(radiobutton, row_counter)
               add_centered_widget(grid, radiobutton, row, 0)

               model_name_label = QLabel()
               model_name_label.setTextFormat(Qt.RichText)
               model_name_label.setText(f'<a style="color: #00bf9e" href="https://huggingface.co/{model["repo_id"]}">{model["name"]}</a>')
               model_name_label.setOpenExternalLinks(False)
               model_name_label.linkActivated.connect(self.open_link)
               model_name_label.setToolTip(TOOLTIPS.get("VECTOR_MODEL_NAME", ""))
               add_centered_widget(grid, model_name_label, row, 1)

               precision_label = QLabel(str(model.get('precision', 'N/A')))
               precision_label.setToolTip(TOOLTIPS.get("VECTOR_MODEL_PRECISION", ""))
               add_centered_widget(grid, precision_label, row, 2)

               parameters_label = QLabel(str(model.get('parameters', 'N/A')))
               parameters_label.setToolTip(TOOLTIPS.get("VECTOR_MODEL_PARAMETERS", ""))
               add_centered_widget(grid, parameters_label, row, 3)

               dimensions_label = QLabel(str(model['dimensions']))
               dimensions_label.setToolTip(TOOLTIPS.get("VECTOR_MODEL_DIMENSIONS", ""))
               add_centered_widget(grid, dimensions_label, row, 4)

               max_sequence_label = QLabel(str(model['max_sequence']))
               max_sequence_label.setToolTip(TOOLTIPS.get("VECTOR_MODEL_MAX_SEQUENCE", ""))
               add_centered_widget(grid, max_sequence_label, row, 5)

               size_label = QLabel(str(model['size_mb']))
               size_label.setToolTip(TOOLTIPS.get("VECTOR_MODEL_SIZE", ""))
               add_centered_widget(grid, size_label, row, 6)

               if 'cache_dir' in model:
                   expected_dir_name = model['cache_dir']
               else:
                   expected_dir_name = ModelDownloader(model_info, model['type']).get_model_directory_name()
               
               is_downloaded = expected_dir_name in existing_vector_directories
               downloaded_label = QLabel('Yes' if is_downloaded else 'No')
               downloaded_label.setToolTip(TOOLTIPS.get("VECTOR_MODEL_DOWNLOADED", ""))
               add_centered_widget(grid, downloaded_label, row, 7)
               radiobutton.setEnabled(not is_downloaded)

               self.downloaded_labels[f"{vendor}/{model['name']}"] = (downloaded_label, model_info, radiobutton)

               row_counter += 1

       for vendor, group_box in self.group_boxes.items():
           self.main_layout.addWidget(group_box)

       self.download_button = QPushButton('Download Selected Model')
       self.download_button.setToolTip(TOOLTIPS.get("DOWNLOAD_MODEL", ""))
       self.download_button.clicked.connect(self.initiate_model_download)
       self.main_layout.addWidget(self.download_button)

       model_downloaded_signal.downloaded.connect(self.update_model_downloaded_status)

    def initiate_model_download(self):
       selected_id = self.model_radiobuttons.checkedId()
       if selected_id != -1:
           _, model_info, _ = list(self.downloaded_labels.values())[selected_id - 1]
           model_downloader = ModelDownloader(model_info, model_info['type'])

           download_thread = threading.Thread(target=lambda: model_downloader.download())
           download_thread.start()

    def update_model_downloaded_status(self, model_name, model_type):
       models_dir = Path('Models')
       vector_models_dir = models_dir / "vector"

       existing_vector_directories = {d.name for d in vector_models_dir.iterdir() if d.is_dir()}

       for vendor, models in VECTOR_MODELS.items():
           for model in models:
               cache_dir = model.get('cache_dir', '')
               generated_dir = model['repo_id'].replace('/', '--')
               
               if cache_dir == model_name or generated_dir == model_name:
                   key = f"{vendor}/{model['name']}"
                   if key in self.downloaded_labels:
                       downloaded_label, _, radiobutton = self.downloaded_labels[key]
                       downloaded_label.setText('Yes')
                       radiobutton.setEnabled(False)
                   self.refresh_gui()
                   return
       
       print(f"Model {model_name} not found in VECTOR_MODELS")

    def refresh_gui(self):
       for group_box in self.group_boxes.values():
           group_box.repaint()
       self.repaint()

    def open_link(self, url):
        QDesktopServices.openUrl(QUrl(url))

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    window = VectorModelsTab()
    window.show()
    app.exec()
