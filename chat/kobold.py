import json
import logging
import requests
import sseclient
from PySide6.QtCore import QThread, Signal

from db.database_interactions import QueryVectorDB
from chat.base import ChatSignals, load_chat_config, save_metadata, build_augmented_query, write_chat_history, cleanup_gpu
from core.utilities import format_citations
from core.constants import PROJECT_ROOT

class KoboldChat:
    def __init__(self):
        self.signals = ChatSignals()
        self.config = load_chat_config()
        self.query_vector_db = None
        self.api_url = "http://localhost:5001/api/extra/generate/stream"
        self.stop_request = False

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
        cleanup_gpu()
        return citations

    def ask_kobold(self, query, selected_database):
        if self.query_vector_db is None or self.query_vector_db.selected_database != selected_database:
            self.query_vector_db = QueryVectorDB.get_instance(selected_database)

        contexts, metadata_list = self.query_vector_db.search(query)
        save_metadata(metadata_list)

        if not contexts:
            self.signals.error_signal.emit("No relevant contexts found.")
            self.signals.finished_signal.emit()
            return

        augmented_query = build_augmented_query(contexts, query)

        full_response = ""
        try:
            response_generator = self.connect_to_kobold(augmented_query)
            for response_chunk in response_generator:
                if self.stop_request:
                    break
                self.signals.response_signal.emit(response_chunk)
                full_response += response_chunk

            write_chat_history(full_response)
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
