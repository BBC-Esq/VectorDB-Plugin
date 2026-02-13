# chat_openai.py
import gc
import logging
from pathlib import Path
import os
import torch
import yaml
from openai import OpenAI
from PySide6.QtCore import QThread, Signal, QObject

from database_interactions import QueryVectorDB
from utilities import format_citations, normalize_chat_text
from constants import rag_string, system_message

ROOT_DIRECTORY = Path(__file__).resolve().parent

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"
chat_history_output_file_path = ROOT_DIRECTORY / "chat_history.txt"


class ChatGPTSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)


class ChatGPTChat:
    def __init__(self, override_model: str = None):
        self.response_callback = lambda x: None
        self.error_callback = lambda x: None
        self.finished_callback = lambda: None
        self.citations_callback = lambda x: None
        self.config = self.load_configuration()
        if override_model:
            self.config.setdefault('openai', {})['model'] = override_model
        self.query_vector_db = None

    def load_configuration(self):
        config_path = ROOT_DIRECTORY / 'config.yaml'
        with config_path.open('r', encoding='utf-8') as config_file:
            return yaml.safe_load(config_file)

    def connect_to_chatgpt(self, augmented_query):
        openai_config = self.config.get('openai', {})
        model = openai_config.get('model', 'gpt-4.1-nano')
        """
        +--------------+--------+--------+-------------+------------+
        |     Model    | Input  | Output | Max Context | Max Output |
        +--------------+--------+--------+-------------+------------+
        | gpt-4.1-nano | $0.10  | $0.40  | 1,047,576   | 32,768     |
        | gpt-4o-mini  | $0.15  | $0.60  | 128,000     | 16,384     |
        | gpt-4.1-mini | $0.40  | $1.60  | 1,047,576   | 32,768     |
        | 04-mini      | $1.10  | $4.40  | 200,000     | 100,000    |
        | gpt-4.1      | $2.00  | $8.00  | 1,047,576   | 32,768     |
        | o3           | $2.00  | $8.00  | 200,000     | 100,000    |
        | gpt-4o       | $2.50  | $10.00 | 128,000     | 16,384     |
        | o3-pro       | $20.00 | $80.00 | 200,000     | 100,000    |
        +--------------+--------+--------+-------------+------------+
        """
        reasoning_effort = openai_config.get('reasoning_effort', 'medium')
        api_key = openai_config.get('api_key')

        if not api_key:
            raise ValueError("OpenAI API key not found in config.yaml.\n\n  Please set it within the 'File' menu.")

        client = OpenAI(api_key=api_key)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": augmented_query}
        ]

        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "stream": True
        }

        # only for thinking models
        # see here before implementing: https://platform.openai.com/docs/guides/reasoning
        if model in ["o4-mini", "o3", "o3-pro"]:
            completion_params["reasoning_effort"] = reasoning_effort

        stream = client.chat.completions.create(**completion_params)

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def handle_response_and_cleanup(self, full_response, metadata_list):
        citations = format_citations(metadata_list)

        if self.query_vector_db:
            if hasattr(self.query_vector_db.embeddings, 'client'):
                del self.query_vector_db.embeddings.client
            del self.query_vector_db.embeddings

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return citations

    def save_metadata_to_file(self, metadata_list):
        with metadata_output_file_path.open('w', encoding='utf-8') as output_file:
            for metadata in metadata_list:
                output_file.write(f"{metadata}\n")

    def ask_chatgpt(self, query, selected_database):
        if self.query_vector_db is None or self.query_vector_db.selected_database != selected_database:
            self.query_vector_db = QueryVectorDB.get_instance(selected_database)

        contexts, metadata_list = self.query_vector_db.search(query)

        self.save_metadata_to_file(metadata_list)
        
        if not contexts:
            self.error_callback("No relevant contexts found.")
            self.finished_callback()
            return

        augmented_query = f"{rag_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + f"\n\n-----\n\n{query}"

        full_response = ""
        response_generator = self.connect_to_chatgpt(augmented_query)
        for response_chunk in response_generator:
            self.response_callback(response_chunk)
            full_response += response_chunk

        with chat_history_output_file_path.open('w', encoding='utf-8') as f:
            normalized_response = normalize_chat_text(full_response)
            f.write(normalized_response)

        self.response_callback("\n")

        citations = self.handle_response_and_cleanup(full_response, metadata_list)
        self.citations_callback(citations)
        self.finished_callback()


class ChatGPTThread(QThread):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)

    def __init__(self, query, selected_database, model_name: str = None):
        super().__init__()
        self.query = query
        self.selected_database = selected_database
        self.chatgpt_chat = ChatGPTChat(override_model=model_name)

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