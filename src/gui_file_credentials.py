from pathlib import Path
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QVBoxLayout, 
                              QLabel, QLineEdit, QPushButton, QMessageBox)
import yaml
import logging
import traceback
from utilities import my_cprint
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class CredentialManager(ABC):
    def __init__(self, parent_widget):
        self.parent_widget = parent_widget
        self.config_file_path = Path(__file__).parent / 'config.yaml'
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if self.config_file_path.exists():
            with open(self.config_file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        return {}

    def _save_config(self) -> None:
        with open(self.config_file_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(self.config, file, allow_unicode=True)

    @property
    @abstractmethod
    def dialog_title(self) -> str:
        pass

    @property
    @abstractmethod
    def dialog_label(self) -> str:
        pass

    @property
    @abstractmethod
    def clear_button_text(self) -> str:
        pass

    @property
    @abstractmethod
    def credential_name(self) -> str:
        pass

    @abstractmethod
    def get_current_credential(self) -> Optional[str]:
        pass

    @abstractmethod
    def update_credential(self, value: Optional[str]) -> None:
        pass

    def show_dialog(self) -> None:
        try:
            dialog = QDialog(self.parent_widget)
            dialog.setWindowTitle(self.dialog_title)

            layout = QVBoxLayout(dialog)
            
            label = QLabel(self.dialog_label, dialog)
            layout.addWidget(label)

            credential_input = QLineEdit(dialog)
            current_value = self.get_current_credential()
            if current_value:
                credential_input.setText(current_value)
            layout.addWidget(credential_input)

            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            clear_button = QPushButton(self.clear_button_text)
            button_box.addButton(clear_button, QDialogButtonBox.ActionRole)
            layout.addWidget(button_box)

            def save_credential():
                if credential := credential_input.text():
                    self.update_credential(credential)
                    self._save_config()
                    QMessageBox.information(self.parent_widget, "Success", 
                                         f"{self.credential_name} saved successfully.")
                    my_cprint(f"{self.credential_name} updated successfully.", "green")
                dialog.accept()

            def clear_credential():
                self.update_credential(None)
                self._save_config()
                QMessageBox.information(self.parent_widget, "Success", 
                                      f"{self.credential_name} cleared successfully.")
                my_cprint(f"{self.credential_name} cleared.", "green")
                dialog.accept()

            button_box.accepted.connect(save_credential)
            button_box.rejected.connect(dialog.reject)
            clear_button.clicked.connect(clear_credential)

            dialog.exec()

        except Exception as e:
            logging.error(f"Error managing {self.credential_name}: {str(e)}")
            logging.debug(traceback.format_exc())
            QMessageBox.critical(self.parent_widget, "Error", 
                               f"Failed to manage {self.credential_name}: {str(e)}")

class HuggingFaceCredentialManager(CredentialManager):
    @property
    def dialog_title(self) -> str:
        return "Hugging Face Access Token"

    @property
    def dialog_label(self) -> str:
        return "Enter a new Hugging Face access token or clear the current one:"

    @property
    def clear_button_text(self) -> str:
        return "Clear Token"

    @property
    def credential_name(self) -> str:
        return "Hugging Face access token"

    def get_current_credential(self) -> Optional[str]:
        return self.config.get('hf_access_token')

    def update_credential(self, value: Optional[str]) -> None:
        self.config['hf_access_token'] = value

class OpenAICredentialManager(CredentialManager):
    @property
    def dialog_title(self) -> str:
        return "OpenAI API Key"

    @property
    def dialog_label(self) -> str:
        return "Enter a new OpenAI API key or clear the current one:"

    @property
    def clear_button_text(self) -> str:
        return "Clear Key"

    @property
    def credential_name(self) -> str:
        return "OpenAI API key"

    def get_current_credential(self) -> Optional[str]:
        return self.config.get('openai', {}).get('api_key')

    def update_credential(self, value: Optional[str]) -> None:
        if 'openai' not in self.config:
            self.config['openai'] = {}
        self.config['openai']['api_key'] = value

def manage_credentials(parent_widget, credential_type: str) -> None:
    managers = {
        'hf': HuggingFaceCredentialManager,
        'openai': OpenAICredentialManager
    }
    
    if manager_class := managers.get(credential_type):
        manager = manager_class(parent_widget)
        manager.show_dialog()
    else:
        raise ValueError(f"Unknown credential type: {credential_type}")