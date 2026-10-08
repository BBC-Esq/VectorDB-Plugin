import yaml
from PySide6.QtCore import QObject, Signal

from core.constants import PROJECT_ROOT


class ChatSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)

def load_chat_config():
    with open(PROJECT_ROOT / 'config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
