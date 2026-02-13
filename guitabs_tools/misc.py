from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QMessageBox, QSpinBox
from PySide6.QtCore import QThread, Signal, QTimer
from initialize import restore_vector_db_backup
from utilities import backup_database

from constants import CustomButtonStyles

class WorkerThread(QThread):
   finished = Signal(bool)

   def __init__(self, function, *args, **kwargs):
       super().__init__()
       self.function = function
       self.args = args
       self.kwargs = kwargs

   def run(self):
       try:
           self.function(*self.args, **self.kwargs)
           self.finished.emit(True)
       except Exception as e:
           print(f"Error during {self.function.__name__}: {e}")
           self.finished.emit(False)

class MiscTab(QWidget):
   def __init__(self):
       super().__init__()
       self.layout = QVBoxLayout(self)
       
       self.backup_all_button = QPushButton("Backup Databases")
       self.backup_all_button.clicked.connect(self.backup_all_databases)
       self.backup_all_button.setToolTip("Create a backup of all databases in the Vector_DB folder")
       
       self.restore_backup_button = QPushButton("Restore Databases")
       self.restore_backup_button.clicked.connect(self.restore_backup)
       self.restore_backup_button.setToolTip("Restore databases from the most recent backup")
       
       self.chart_gpus_button = QPushButton("GPUs")
       self.chart_gpus_button.clicked.connect(self.chart_gpus)
       self.chart_gpus_button.setToolTip("Compare GPUs by V-RAM")
       
       self.chart_chat_models_button = QPushButton("Chat Models")
       self.chart_chat_models_button.clicked.connect(self.chart_chat_models)
       self.chart_chat_models_button.setToolTip("Compare various chat models.")

       self.chart_vision_models_button = QPushButton("Vision Models")
       self.chart_vision_models_button.clicked.connect(self.chart_vision_models)
       self.chart_vision_models_button.setToolTip("Compare various vision models.")

       self.chart_vector_models_button = QPushButton("Vector Models")
       self.chart_vector_models_button.clicked.connect(self.chart_vector_models)
       self.chart_vector_models_button.setToolTip("Compare various vector/embedding models.")

       self.min_vram_spin = QSpinBox()
       self.min_vram_spin.setRange(1, 128)
       self.min_vram_spin.setValue(8)
       self.min_vram_spin.setPrefix("Min ")
       self.min_vram_spin.setSuffix(" GB")
       self.min_vram_spin.setToolTip("Minimum GPU V-RAM (in GB)")

       self.max_vram_spin = QSpinBox()
       self.max_vram_spin.setRange(1, 128)
       self.max_vram_spin.setValue(16)
       self.max_vram_spin.setPrefix("Max ")
       self.max_vram_spin.setSuffix(" GB")
       self.max_vram_spin.setToolTip("Maximum GPU V-RAM (in GB)")
       
       self.backup_all_button.setStyleSheet(CustomButtonStyles.RED_BUTTON_STYLE)
       self.restore_backup_button.setStyleSheet(CustomButtonStyles.RED_BUTTON_STYLE)
       self.chart_gpus_button.setStyleSheet(CustomButtonStyles.GREEN_BUTTON_STYLE)
       self.chart_chat_models_button.setStyleSheet(CustomButtonStyles.BLUE_BUTTON_STYLE)
       self.chart_vision_models_button.setStyleSheet(CustomButtonStyles.TEAL_BUTTON_STYLE)
       self.chart_vector_models_button.setStyleSheet(CustomButtonStyles.PURPLE_BUTTON_STYLE)

       center_button_layout = QHBoxLayout()
       center_button_layout.addStretch(1)
       center_button_layout.addWidget(self.backup_all_button)
       center_button_layout.addWidget(self.restore_backup_button)
       center_button_layout.addWidget(self.chart_gpus_button)
       center_button_layout.addWidget(self.min_vram_spin)
       center_button_layout.addWidget(self.max_vram_spin)
       center_button_layout.addWidget(self.chart_chat_models_button)
       center_button_layout.addWidget(self.chart_vision_models_button)
       center_button_layout.addWidget(self.chart_vector_models_button)
       center_button_layout.addStretch(1)
       
       self.layout.addLayout(center_button_layout)
       
       self.backup_thread = None
       self.restore_thread = None

   def set_buttons_enabled(self, enabled, buttons):
       for button in buttons:
           button.setEnabled(enabled)

   def set_button_text(self, button: QPushButton, text: str):
       button.setText(text)
   
   def backup_all_databases(self):
       confirm = QMessageBox.question(
           self,
           "Confirm Backup",
           "Warning. This will erase any existing backups and overwrite them with the current state of the \"Vector_DB\" folder.\n\nAre you sure you want to proceed?",
           QMessageBox.Yes | QMessageBox.No,
           QMessageBox.No
       )
       
       if confirm == QMessageBox.Yes:
           self.set_button_text(self.backup_all_button, "Backing up...")
           self.set_buttons_enabled(False, [self.backup_all_button, self.restore_backup_button])
           
           self.backup_thread = WorkerThread(backup_database)
           self.backup_thread.finished.connect(self.on_backup_finished)
           self.backup_thread.start()
       else:
           pass

   def on_backup_finished(self, success):
       self.set_buttons_enabled(True, [self.backup_all_button, self.restore_backup_button])
       self.set_button_text(self.backup_all_button, "Backup Databases")
       if success:
           QMessageBox.information(self, "Backup Complete", "All databases have been successfully backed up.")
       else:
           QMessageBox.critical(self, "Backup Failed", "Failed to backup the databases. Check the console for error details.")

   def restore_backup(self):
       confirm = QMessageBox.question(
           self,
           "Confirm Restoration",
           "Warning. This will overwrite current databases with the backup. Are you sure you want to proceed?",
           QMessageBox.Yes | QMessageBox.No,
           QMessageBox.No
       )
       
       if confirm == QMessageBox.Yes:
           self.set_button_text(self.restore_backup_button, "Restoring...")
           self.set_buttons_enabled(False, [self.restore_backup_button, self.backup_all_button])

           self.restore_thread = WorkerThread(restore_vector_db_backup)
           self.restore_thread.finished.connect(self.on_restore_finished)
           self.restore_thread.start()
       else:
           pass

   def on_restore_finished(self, success):
       self.set_buttons_enabled(True, [self.restore_backup_button, self.backup_all_button])
       self.set_button_text(self.restore_backup_button, "Restore Databases")
       if success:
           QMessageBox.information(self, "Restoration Complete", "The databases have been successfully restored from the backup.")
       else:
           QMessageBox.critical(self, "Restoration Failed", "Failed to restore the database backup. Check the console for error details.")

   def chart_gpus(self):
       import matplotlib
       matplotlib.use('QtAgg')
       import matplotlib.pyplot as plt
       from chart_all_gpus import create_gpu_comparison_plot
       
       self.chart_gpus_button.setEnabled(False)
       self.set_button_text(self.chart_gpus_button, "Charting...")

       min_vram = self.min_vram_spin.value()
       max_vram = self.max_vram_spin.value()
       if min_vram > max_vram:
           QMessageBox.warning(self, "Invalid Range", "Minimum V-RAM value cannot exceed maximum V-RAM value.")
           self.reset_chart_button()
           return

       fig = create_gpu_comparison_plot(min_vram, max_vram)
       plt.figure(fig.number)
       plt.show(block=False)

       QTimer.singleShot(500, self.reset_chart_button)

   def reset_chart_button(self):
       self.set_button_text(self.chart_gpus_button, "GPUs")
       self.chart_gpus_button.setEnabled(True)

   def chart_chat_models(self):
       import matplotlib
       matplotlib.use('QtAgg')
       import matplotlib.pyplot as plt
       from chart_models_chat import create_chat_models_comparison_plot
       
       self.chart_chat_models_button.setEnabled(False)
       self.set_button_text(self.chart_chat_models_button, "Charting...")
       
       fig = create_chat_models_comparison_plot()
       plt.figure(fig.number)
       plt.show(block=False)

       QTimer.singleShot(500, self.reset_chart_chat_models_button)

   def reset_chart_chat_models_button(self):
       self.set_button_text(self.chart_chat_models_button, "Chat Models")
       self.chart_chat_models_button.setEnabled(True)

   def chart_vision_models(self):
       import matplotlib
       matplotlib.use('QtAgg')
       import matplotlib.pyplot as plt
       from chart_models_vision import create_vision_models_comparison_plot
       
       self.chart_vision_models_button.setEnabled(False)
       self.set_button_text(self.chart_vision_models_button, "Charting...")
       
       fig = create_vision_models_comparison_plot()
       plt.figure(fig.number)
       plt.show(block=False)

       QTimer.singleShot(500, self.reset_chart_vision_models_button)

   def reset_chart_vision_models_button(self):
       self.set_button_text(self.chart_vision_models_button, "Vision Models")
       self.chart_vision_models_button.setEnabled(True)

   def chart_vector_models(self):
       import matplotlib
       matplotlib.use('QtAgg')
       import matplotlib.pyplot as plt
       from chart_models_vector import create_vector_models_comparison_plot
       
       self.chart_vector_models_button.setEnabled(False)
       self.set_button_text(self.chart_vector_models_button, "Charting...")
       
       fig = create_vector_models_comparison_plot()
       plt.figure(fig.number)
       plt.show(block=False)

       QTimer.singleShot(500, self.reset_chart_vector_models_button)

   def reset_chart_vector_models_button(self):
       self.set_button_text(self.chart_vector_models_button, "Vector Models")
       self.chart_vector_models_button.setEnabled(True)
