import os
import sys
import yaml
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QCheckBox, QPushButton, QWidget, QScrollArea, QLabel, 
                               QSpinBox, QGroupBox)
from PySide6.QtCore import Qt

class FileCompilerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_file = "settings.yaml"
        self.checkboxes = {}
        self.excluded_folders = {'lib', 'venv', 'env', '__pycache__', '.git', '.vscode', 
                               'node_modules', 'build', 'dist', 'site-packages'}
        self.excluded_files = {'__init__.py'}
        self.scan_depth = 2
        self.current_directory = Path.cwd()
        
        self.file_paths = self._scan_python_files()
        self.root_dir = str(self.current_directory)
        self.project_name = self.current_directory.name
        
        self.init_ui()
        self.load_config()

    def _scan_python_files(self):
        python_files = []
        
        def should_exclude_folder(folder_name):
            return folder_name.lower() in self.excluded_folders
        
        def should_exclude_file(file_path):
            file_name = file_path.name.lower()
            return file_name in {f.lower() for f in self.excluded_files}
        
        def scan_directory(directory, current_depth, max_depth):
            try:
                for item in directory.iterdir():
                    if item.is_file() and item.suffix.lower() == '.py':
                        if not should_exclude_file(item):
                            python_files.append(str(item.resolve()))
                    elif item.is_dir() and current_depth < max_depth:
                        if not should_exclude_folder(item.name):
                            scan_directory(item, current_depth + 1, max_depth)
            except PermissionError:
                pass
        
        scan_directory(self.current_directory, 0, self.scan_depth)
        return sorted(python_files)

    def _find_common_root(self):
        if not self.file_paths:
            return str(self.current_directory)

        paths = [Path(p) for p in self.file_paths]
        
        return str(self.current_directory)

    def _get_relative_path(self, file_path):
        try:
            rel_path = Path(file_path).relative_to(self.current_directory)
            return str(rel_path).replace('\\', '/')
        except ValueError:
            return file_path

    def _get_file_char_count(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return len(content)
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin1') as file:
                    content = file.read()
                    return len(content)
            except Exception:
                return 0
        except Exception:
            return 0

    def _calculate_total_chars(self):
        total_chars = 0
        selected_count = 0
        
        for file_path, checkbox in self.checkboxes.items():
            if checkbox.isChecked():
                selected_count += 1
                total_chars += self._get_file_char_count(file_path)
        
        return selected_count, total_chars

    def _update_total_display(self):
        selected_count, total_chars = self._calculate_total_chars()
        self.total_chars_label.setText(f"Selected: {selected_count} files, {total_chars:,} characters")

    def init_ui(self):
        self.setWindowTitle("Python File Compiler")
        self.setGeometry(100, 100, 900, 1100)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        project_info = QLabel(f"Project: {self.project_name}\nRoot: {self.current_directory}")
        project_info.setStyleSheet("font-weight: bold; font-size: 12px; margin: 10px; padding: 10px; border-radius: 5px;")
        layout.addWidget(project_info)

        scan_group = QGroupBox("Scan Settings")
        scan_layout = QHBoxLayout(scan_group)

        depth_label = QLabel("Scan Depth:")
        self.depth_spinbox = QSpinBox()
        self.depth_spinbox.setMinimum(0)
        self.depth_spinbox.setMaximum(10)
        self.depth_spinbox.setValue(self.scan_depth)
        self.depth_spinbox.setToolTip("0 = current directory only, 1 = include subfolders, etc.")
        
        refresh_btn = QPushButton("Refresh Scan")
        refresh_btn.clicked.connect(self.refresh_scan)
        refresh_btn.setToolTip("Rescan for Python files with current depth setting")

        scan_layout.addWidget(depth_label)
        scan_layout.addWidget(self.depth_spinbox)
        scan_layout.addWidget(refresh_btn)
        scan_layout.addStretch()

        layout.addWidget(scan_group)

        self.file_count_label = QLabel(f"Found {len(self.file_paths)} Python files")
        self.file_count_label.setStyleSheet("font-size: 12px; margin: 5px;")
        layout.addWidget(self.file_count_label)

        title_label = QLabel("Select files to include in compilation:")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        layout.addWidget(title_label)

        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        button_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(deselect_all_btn)

        copy_btn = QPushButton("Copy Selected Files to Clipboard")
        copy_btn.clicked.connect(self.copy_to_clipboard)
        copy_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white; padding: 10px;")
        button_layout.addWidget(copy_btn)

        layout.addLayout(button_layout)

        excluded_info = QLabel(f"Excluded folders: {', '.join(sorted(self.excluded_folders))}\n"
                              f"Excluded files: {', '.join(sorted(self.excluded_files))}")
        excluded_info.setStyleSheet("font-size: 10px; color: gray; margin: 5px;")
        excluded_info.setWordWrap(True)
        layout.addWidget(excluded_info)

        self.total_chars_label = QLabel("Selected: 0 files, 0 characters")
        self.total_chars_label.setStyleSheet("font-size: 12px; margin: 5px; font-weight: bold; color: #2196F3;")
        layout.addWidget(self.total_chars_label)

        self._populate_file_list()

    def _populate_file_list(self):
        for checkbox in self.checkboxes.values():
            checkbox.setParent(None)
        self.checkboxes.clear()

        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for file_path in self.file_paths:
            rel_path = self._get_relative_path(file_path)
            char_count = self._get_file_char_count(file_path)
            display_text = f"{rel_path} ({char_count:,} chars)"
            
            checkbox = QCheckBox(display_text)
            checkbox.setChecked(True)
            checkbox.setToolTip(file_path)
            checkbox.stateChanged.connect(self._update_total_display)
            self.checkboxes[file_path] = checkbox
            self.scroll_layout.addWidget(checkbox)

        self._update_total_display()

    def refresh_scan(self):
        self.scan_depth = self.depth_spinbox.value()
        self.file_paths = self._scan_python_files()
        self.root_dir = str(self.current_directory)
        
        self.file_count_label.setText(f"Found {len(self.file_paths)} Python files")
        
        self._populate_file_list()
        
        self.load_config()

    def select_all(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)
        self._update_total_display()

    def deselect_all(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)
        self._update_total_display()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file) or {}
                for file_path, checkbox in self.checkboxes.items():
                    if file_path in config:
                        checkbox.setChecked(config[file_path])
        except FileNotFoundError:
            pass
        
        self._update_total_display()

    def save_config(self):
        config = {}
        for file_path, checkbox in self.checkboxes.items():
            config[file_path] = checkbox.isChecked()
        
        config['_scan_depth'] = self.scan_depth
        
        with open(self.config_file, 'w') as file:
            yaml.dump(config, file)

    def copy_to_clipboard(self):
        selected_files = [file_path for file_path, checkbox in self.checkboxes.items() if checkbox.isChecked()]
        
        if not selected_files:
            return

        compiled_text = f"# {self.project_name} - Source Code\n\n"
        compiled_text += f"Project structure with {len(selected_files)} files:\n\n"

        file_groups = {}
        for file_path in selected_files:
            rel_path = self._get_relative_path(file_path)
            directory = str(Path(rel_path).parent).replace('\\', '/') if Path(rel_path).parent != Path('.') else 'root'
            if directory not in file_groups:
                file_groups[directory] = []
            file_groups[directory].append((file_path, rel_path))

        for directory, files in sorted(file_groups.items()):
            compiled_text += f"## {directory}/\n"
            for _, rel_path in files:
                compiled_text += f"- {Path(rel_path).name}\n"
            compiled_text += "\n"

        compiled_text += "---\n\n"

        for i, file_path in enumerate(selected_files):
            rel_path = self._get_relative_path(file_path)
            compiled_text += f"## File: {rel_path}\n\n"
            compiled_text += "```python\n"

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    compiled_text += file_content
            except FileNotFoundError:
                compiled_text += f"# ERROR: File not found - {file_path}"
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin1') as file:
                        file_content = file.read()
                        compiled_text += file_content
                except Exception as e:
                    compiled_text += f"# ERROR reading {file_path}: {str(e)}"
            except Exception as e:
                compiled_text += f"# ERROR reading {file_path}: {str(e)}"
            
            compiled_text += "\n```\n\n"

        try:
            app = QApplication.instance()
            clipboard = app.clipboard()
            clipboard.setText(compiled_text)

            self.save_config()

        except Exception as e:
            with open('compiled_files.txt', 'w', encoding='utf-8') as f:
                f.write(compiled_text)

    def closeEvent(self, event):
        self.save_config()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = FileCompilerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()