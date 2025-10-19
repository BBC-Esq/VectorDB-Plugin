import gc
import json
import logging
from pathlib import Path
import torch
import yaml
import requests
import sseclient
from PySide6.QtCore import QThread, Signal, QObject

from database_interactions import QueryVectorDB
from utilities import format_citations, normalize_chat_text
from constants import rag_string

ROOT_DIRECTORY = Path(__file__).resolve().parent

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"

class KoboldSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)

class KoboldChat:
    def __init__(self):
        self.signals = KoboldSignals()
        self.config = self.load_configuration()
        self.query_vector_db = None
        self.api_url = "http://localhost:5001/api/extra/generate/stream"
        self.stop_request = False

    def load_configuration(self):
        config_path = ROOT_DIRECTORY / 'config.yaml'
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def connect_to_kobold(self, augmented_query):
        payload = {
            "prompt": augmented_query,
            "max_context_length": 8192,
            "max_length": 1024,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        response = None
        try:
            response = requests.post(self.api_url, json=payload, stream=True, timeout=20)
            response.raise_for_status()
            client = sseclient.SSEClient(response)

            for event in client.events():
                if self.stop_request:
                    break
                if event.event == "message":
                    try:
                        data = json.loads(event.data)
                        if 'token' in data:
                            yield data['token']
                    except json.JSONDecodeError:
                        logging.error(f"Failed to parse JSON: {event.data}")
                        raise ValueError(f"Failed to parse response: {event.data}")
        except Exception as e:
            logging.error(f"Error in Kobold API request: {str(e)}")
            raise
        finally:
            if response:
                response.close()

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

    def ask_kobold(self, query, selected_database):
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
        try:
            response_generator = self.connect_to_kobold(augmented_query)
            for response_chunk in response_generator:
                if self.stop_request:
                    break
                self.signals.response_signal.emit(response_chunk)
                full_response += response_chunk

            chat_history_path = ROOT_DIRECTORY / 'chat_history.txt'
            with open(chat_history_path, 'w', encoding='utf-8') as f:
                normalized_response = normalize_chat_text(full_response)
                f.write(normalized_response)

            self.signals.response_signal.emit("\n")

            citations = self.handle_response_and_cleanup(full_response, metadata_list)
            self.signals.citations_signal.emit(citations)
        except Exception as e:
            self.signals.error_signal.emit(str(e))
            raise

class KoboldThread(QThread):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)
    
    def __init__(self, query, selected_database):
        super().__init__()
        self.query = query
        self.selected_database = selected_database
        self.kobold_chat = KoboldChat()
        self.kobold_chat.signals.response_signal.connect(self.response_signal.emit)
        self.kobold_chat.signals.error_signal.connect(self.error_signal.emit)
        self.kobold_chat.signals.citations_signal.connect(self.citations_signal.emit)

    def run(self):
        try:
            self.kobold_chat.ask_kobold(self.query, self.selected_database)
        except Exception as e:
            logging.error(f"Error in KoboldThread: {str(e)}")
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()
            
    def stop(self):
        self.kobold_chat.stop_request = True
        self.wait(5000)