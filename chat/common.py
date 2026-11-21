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


NO_CHUNKS = (
    "No chunks passed the similarity threshold. "
    "Try lowering the 'Similarity' setting in the Database Query settings tab."
)


def no_chunks_message():
    try:
        search_term = str(((load_chat_config() or {}).get('database') or {}).get('search_term') or '').strip()
    except Exception:
        search_term = ''
    if not search_term:
        return NO_CHUNKS
    return (
        f'No chunks that passed the similarity threshold contain the search term "{search_term}". '
        "Clear or change the Search term on the Settings tab, or lower the Similarity setting."
    )
