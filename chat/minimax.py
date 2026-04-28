import logging
from openai import OpenAI
from PySide6.QtCore import QThread, Signal

from db.database_interactions import get_query_db
from chat.base import load_chat_config, save_metadata, build_augmented_query, cleanup_gpu
from core.utilities import format_citations
from core.constants import system_message, PROJECT_ROOT

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODELS = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]
# Temperature must be in (0.0, 1.0] for MiniMax
_MINIMAX_MIN_TEMP = 0.01


class MiniMaxChat:
    def __init__(self, override_model: str = None):
        self.response_callback = lambda x: None
        self.error_callback = lambda x: None
        self.finished_callback = lambda: None
        self.citations_callback = lambda x: None
        self.config = load_chat_config()
        if override_model:
            self.config.setdefault('minimax', {})['model'] = override_model
        self.query_vector_db = None

    def connect_to_minimax(self, augmented_query):
        minimax_config = self.config.get('minimax', {})
        model = minimax_config.get('model', 'MiniMax-M2.7')
        api_key = minimax_config.get('api_key')

        if not api_key:
            raise ValueError("MiniMax API key not found in config.yaml.\n\n  Please set it within the 'File' menu.")

        client = OpenAI(api_key=api_key, base_url=MINIMAX_BASE_URL)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": augmented_query}
        ]

        # MiniMax temperature must be in (0.0, 1.0]
        temperature = max(_MINIMAX_MIN_TEMP, 0.1)

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def handle_response_and_cleanup(self, full_response, metadata_list):
        citations = format_citations(metadata_list)

        if self.query_vector_db:
            if hasattr(self.query_vector_db.embeddings, 'client'):
                del self.query_vector_db.embeddings.client
            del self.query_vector_db.embeddings

        cleanup_gpu()
        return citations

    def ask_minimax(self, query, selected_database):
        if self.query_vector_db is None or self.query_vector_db.selected_database != selected_database:
            self.query_vector_db = get_query_db(selected_database)

        contexts, metadata_list = self.query_vector_db.search(query)
        save_metadata(metadata_list)

        if not contexts:
            self.error_callback(
                "No chunks passed the similarity threshold. "
                "Try lowering the 'Similarity' setting in the Database Query settings tab."
            )
            self.finished_callback()
            return

        augmented_query = build_augmented_query(contexts, query)

        full_response = ""
        response_generator = self.connect_to_minimax(augmented_query)
        for response_chunk in response_generator:
            self.response_callback(response_chunk)
            full_response += response_chunk

        self.response_callback("\n")

        citations = self.handle_response_and_cleanup(full_response, metadata_list)
        self.citations_callback(citations)
        self.finished_callback()


class MiniMaxThread(QThread):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)

    def __init__(self, query, selected_database, model_name: str = None):
        super().__init__()
        self.query = query
        self.selected_database = selected_database
        self.minimax_chat = MiniMaxChat(override_model=model_name)

        self.minimax_chat.response_callback = self.on_response
        self.minimax_chat.error_callback = self.on_error
        self.minimax_chat.finished_callback = self.on_finished
        self.minimax_chat.citations_callback = self.on_citations

    def on_response(self, text):
        self.response_signal.emit(text)

    def on_error(self, error):
        self.error_signal.emit(error)

    def on_finished(self):
        self.finished_signal.emit()

    def on_citations(self, citations):
        self.citations_signal.emit(citations)

    def run(self):
        try:
            self.minimax_chat.ask_minimax(self.query, self.selected_database)
        except Exception as e:
            logging.error(f"Error in MiniMaxThread: {str(e)}")
            self.on_error(str(e))
        finally:
            self.on_finished()
