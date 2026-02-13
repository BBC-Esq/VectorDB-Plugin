import gc
import logging

import requests
from pathlib import Path
import torch
import yaml
from openai import OpenAI
from PySide6.QtCore import QThread, Signal, QObject

from database_interactions import QueryVectorDB
from utilities import format_citations, normalize_chat_text
from constants import system_message, rag_string, THINKING_TAGS

ROOT_DIRECTORY = Path(__file__).resolve().parent

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"

class LMStudioSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)

class LMStudioChat:
    def __init__(self):
        self.signals = LMStudioSignals()
        self.config = self.load_configuration()
        self.query_vector_db = None

    def load_configuration(self):
        with open('config.yaml', 'r') as config_file:
            return yaml.safe_load(config_file)

    def connect_to_local_chatgpt(self, prompt):
        server_config = self.config.get('server', {})
        base_url = server_config.get('connection_str')
        show_thinking = server_config.get('show_thinking', False)

        client = OpenAI(base_url=base_url, api_key='lm-studio')
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        stream = client.chat.completions.create(
            model="local-model",
            messages=messages,
            stream=True
        )

        in_thinking_block = False
        first_content = True
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content

                if not show_thinking:
                    skip_chunk = False
                    # Check each pair of thinking tags
                    for start_tag, end_tag in THINKING_TAGS.values():
                        if start_tag in content:
                            in_thinking_block = True
                            skip_chunk = True
                            break
                        if end_tag in content:
                            in_thinking_block = False
                            skip_chunk = True
                            break
                    if skip_chunk:
                        continue
                    if in_thinking_block:
                        continue

                if first_content:
                    content = content.lstrip()
                    first_content = False

                yield content

    def handle_response_and_cleanup(self, full_response, metadata_list):
        citations = format_citations(metadata_list)
        
        if self.query_vector_db:
            self.query_vector_db.cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return citations

    def save_metadata_to_file(self, metadata_list):
        with metadata_output_file_path.open('w', encoding='utf-8') as output_file:
            for metadata in metadata_list:
                output_file.write(f"{metadata}\n")

    def ask_local_chatgpt(self, query, selected_database):
        if self.query_vector_db is None or self.query_vector_db.selected_database != selected_database:
            self.query_vector_db = QueryVectorDB.get_instance(selected_database)

        contexts, metadata_list = self.query_vector_db.search(query)

        self.save_metadata_to_file(metadata_list)

        if not contexts:
            self.signals.error_signal.emit("No relevant contexts found.")
            self.signals.finished_signal.emit()
            return

        augmented_query = f"{rag_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + f"\n\n-----\n\n{query}"
        
        full_response = ""
        response_generator = self.connect_to_local_chatgpt(augmented_query)
        for response_chunk in response_generator:
            self.signals.response_signal.emit(response_chunk)
            full_response += response_chunk

        with open('chat_history.txt', 'w', encoding='utf-8') as f:
            normalized_response = normalize_chat_text(full_response)
            f.write(normalized_response)

        self.signals.response_signal.emit("\n")

        citations = self.handle_response_and_cleanup(full_response, metadata_list)
        self.signals.citations_signal.emit(citations)
        self.signals.finished_signal.emit()

class LMStudioChatThread(QThread):
    def __init__(self, query, selected_database):
        super().__init__()
        self.query = query
        self.selected_database = selected_database
        self.lm_studio_chat = LMStudioChat()

    def run(self):
        try:
            self.lm_studio_chat.ask_local_chatgpt(self.query, self.selected_database)
        except Exception as e:
            logging.error(f"Error in LMStudioChatThread: {str(e)}")
            self.lm_studio_chat.signals.error_signal.emit(str(e))

def is_lm_studio_available():
    try:
        response = requests.get("http://127.0.0.1:1234/v1/models/", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

"""
[Main Process]
    |
    |         DatabaseQueryTab (GUI)                 LMStudioChatThread
    |         ------------------                     -----------------
    |              |                                     |
    |        [Submit Button]                             |
    |              |                                     |
    |         on_submit_button_clicked()                 |
    |              |                                     |
    |              |---> LMStudioChatThread.start() ---->|
    |              |                                     |
    |                                          [LMStudioChat Instance]
    |                                                    |
    |                                         ask_local_chatgpt()
    |                                                    |
    |                                         [QueryVectorDB Search]
    |                                                    |
    |                                      connect_to_local_chatgpt()
    |                                                    |
    |    Signal Flow                            OpenAI API Stream
    |    -----------                            ----------------
    |         |                                        |
    |    Signals Received:                     Stream Chunks:
    |    - response_signal                     - chunk.choices[0].delta.content
    |    - error_signal                                |
    |    - finished_signal                             |
    |    - citations_signal                             |
    |         |                                        |
    |    GUI Updates:                          Cleanup Operations:
    |    - update_response_lm_studio()         - handle_response_and_cleanup()
    |    - show_error_message()                - save_metadata_to_file()
    |    - on_submission_finished()            - torch.cuda.empty_cache()
    |    - display_citations_in_widget()       - gc.collect()
    |                                                  |
    |                                          Emit Final Signals:
    |                                          - citations_signal
    |                                          - finished_signal
"""