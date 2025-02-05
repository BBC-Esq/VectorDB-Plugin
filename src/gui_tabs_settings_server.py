import yaml
from pathlib import Path

from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QGridLayout, QMessageBox, QSizePolicy, QCheckBox)

from constants import TOOLTIPS

class ServerSettingsTab(QWidget):
    def __init__(self):
        super(ServerSettingsTab, self).__init__()

        try:
            with open('config.yaml', 'r', encoding='utf-8') as file:
                self.config_data = yaml.safe_load(file)
                self.server_config = self.config_data.get('server', {})
                self.connection_str = self.server_config.get('connection_str', '')
                self.show_thinking = self.server_config.get('show_thinking', False)
                if ':' in self.connection_str and '/' in self.connection_str:
                    self.current_port = self.connection_str.split(":")[-1].split("/")[0]
                else:
                    self.current_port = ''
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"An error occurred while loading the configuration: {e}"
            )
            self.server_config = {}
            self.connection_str = ''
            self.current_port = ''
            self.show_thinking = False

        settings_dict = {
            'port': {
                "placeholder": "Port...",
                "validator": QIntValidator(1, 65535),
                "current": self.current_port
            }
        }

        self.widgets = {}
        layout = QGridLayout()

        # port
        port_label = self.create_label('port', settings_dict)
        port_label.setToolTip(TOOLTIPS["PORT"])
        layout.addWidget(port_label, 0, 0)
        port_edit = self.create_edit('port', settings_dict)
        port_edit.setToolTip(TOOLTIPS["PORT"])
        layout.addWidget(port_edit, 0, 1)

        # show Thinking Checkbox
        self.thinking_checkbox = QCheckBox("Show thinking process?")
        self.thinking_checkbox.setChecked(self.show_thinking)
        self.thinking_checkbox.setToolTip(TOOLTIPS["SHOW_THINKING_CHECKBOX"])
        self.thinking_checkbox.stateChanged.connect(self.update_show_thinking)
        layout.addWidget(self.thinking_checkbox, 0, 2)

        self.setLayout(layout)

    def update_show_thinking(self, state):
        try:
            with open('config.yaml', 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            
            config_data['server']['show_thinking'] = bool(state)
            
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.safe_dump(config_data, file)
                
            self.show_thinking = bool(state)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Updating Configuration",
                f"An error occurred while updating show_thinking setting: {e}"
            )

    def create_label(self, setting, settings_dict):
        label_text = f"{setting.replace('_', ' ').capitalize()}: {settings_dict[setting]['current']}"
        label = QLabel(label_text)
        self.widgets[setting] = {"label": label}
        return label

    def create_edit(self, setting, settings_dict):
        edit = QLineEdit()
        edit.setPlaceholderText(settings_dict[setting]['placeholder'])
        edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if settings_dict[setting]['validator']:
            edit.setValidator(settings_dict[setting]['validator'])
        self.widgets[setting]['edit'] = edit
        return edit

    def update_config(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with config_file_path.open('r', encoding='utf-8') as file:
                    config_data = yaml.safe_load(file)
                    self.server_config = config_data.get('server', {})
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Configuration",
                    f"An error occurred while loading the configuration: {e}"
                )
                return False
        else:
            QMessageBox.critical(
                self,
                "Configuration File Missing",
                "The configuration file 'config.yaml' does not exist."
            )
            return False

        settings_changed = False
        errors = []

        new_port_text = self.widgets['port']['edit'].text().strip()
        if new_port_text:
            try:
                new_port = int(new_port_text)
                if not (1 <= new_port <= 65535):
                    raise ValueError("Port must be between 1 and 65535.")
            except ValueError:
                errors.append("Port must be an integer between 1 and 65535.")
        else:
            new_port = self.current_port

        if errors:
            error_message = "\n".join(errors)
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"The following errors occurred:\n{error_message}"
            )
            return False

        if new_port_text and new_port != self.current_port:
            if ':' in self.connection_str and '/' in self.connection_str:
                new_connection_str = self.connection_str.replace(self.current_port, str(new_port))
                config_data['server']['connection_str'] = new_connection_str
                settings_changed = True
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Connection String",
                    "The existing connection string format is invalid. Unable to update port."
                )
                return False

        if settings_changed:
            try:
                with config_file_path.open('w', encoding='utf-8') as file:
                    yaml.safe_dump(config_data, file)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Configuration",
                    f"An error occurred while saving the configuration: {e}"
                )
                return False

            if new_port_text:
                self.connection_str = config_data['server']['connection_str']
                self.current_port = str(new_port)
                self.widgets['port']['label'].setText(f"Port: {new_port}")

            self.widgets['port']['edit'].clear()

        return settings_changed