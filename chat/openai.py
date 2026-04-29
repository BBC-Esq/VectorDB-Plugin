import logging
from openai import OpenAI
from PySide6.QtCore import QThread, Signal

from db.database_interactions import get_query_db
from chat.base import load_chat_config, save_metadata, build_augmented_query, cleanup_gpu
from core.utilities import format_citations
from core.constants import system_message
from core.chatgpt_settings import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_VERBOSITY,
    DEFAULT_REASONING_EFFORT,
    supports_verbosity,
    supports_reasoning_effort,
)


class ChatGPTChat:
    def __init__(self):
        self.response_callback = lambda x: None
        self.error_callback = lambda x: None
        self.finished_callback = lambda: None
        self.citations_callback = lambda x: None
        self.config = load_chat_config()
        self.query_vector_db = None

    def connect_to_chatgpt(self, augmented_query):
        openai_config = self.config.get('openai', {}) or {}
        model = openai_config.get('model') or DEFAULT_OPENAI_MODEL
        api_key = openai_config.get('api_key')
        verbosity = openai_config.get('verbosity') or DEFAULT_VERBOSITY
        reasoning_effort = openai_config.get('reasoning_effort') or DEFAULT_REASONING_EFFORT

        if not api_key:
            raise ValueError(
                "OpenAI API key not found in config.yaml.\n\n"
                "Please set it via File menu → Chat Backend Settings…"
            )

        client = OpenAI(api_key=api_key)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": augmented_query},
        ]

        request_args = {
            "model": model,
            "input": messages,
            "stream": True,
        }

        if supports_verbosity(model):
            request_args["text"] = {"verbosity": verbosity}

        if supports_reasoning_effort(model) and reasoning_effort and reasoning_effort != "none":
            request_args["reasoning"] = {"effort": reasoning_effort}

        stream = client.responses.create(**request_args)

        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    yield delta
            elif event_type == "response.error":
                msg = str(getattr(event, "error", "unknown error"))
                logging.error(f"OpenAI Responses API error: {msg}")
                raise RuntimeError(msg)

    def handle_response_and_cleanup(self, full_response, metadata_list):
        citations = format_citations(metadata_list)

        if self.query_vector_db:
            if hasattr(self.query_vector_db.embeddings, 'client'):
                del self.query_vector_db.embeddings.client
            del self.query_vector_db.embeddings

        cleanup_gpu()
        return citations

    def ask_chatgpt(self, query, selected_database):
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
        response_generator = self.connect_to_chatgpt(augmented_query)
        for response_chunk in response_generator:
            self.response_callback(response_chunk)
            full_response += response_chunk

        self.response_callback("\n")

        citations = self.handle_response_and_cleanup(full_response, metadata_list)
        self.citations_callback(citations)
        self.finished_callback()


class ChatGPTThread(QThread):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)

    def __init__(self, query, selected_database):
        super().__init__()
        self.query = query
        self.selected_database = selected_database
        self.chatgpt_chat = ChatGPTChat()

        self.chatgpt_chat.response_callback = self.on_response
        self.chatgpt_chat.error_callback = self.on_error
        self.chatgpt_chat.finished_callback = self.on_finished
        self.chatgpt_chat.citations_callback = self.on_citations

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
            self.chatgpt_chat.ask_chatgpt(self.query, self.selected_database)
        except Exception as e:
            logging.error(f"Error in ChatGPTThread: {str(e)}")
            self.on_error(str(e))
        finally:
            self.on_finished()
