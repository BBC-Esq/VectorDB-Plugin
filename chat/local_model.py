import time
import logging
from enum import Enum, auto
from typing import Any, Optional
from dataclasses import dataclass

import torch
from multiprocessing import Process, Pipe
from multiprocessing.connection import PipeConnection
from PySide6.QtCore import QObject, Signal

import chat.base as module_chat
from db.database_interactions import QueryVectorDB
from core.utilities import format_citations, my_cprint, normalize_chat_text
from core.constants import rag_string, PROJECT_ROOT
from pathlib import Path

class MessageType(Enum):
    QUESTION = auto()
    RESPONSE = auto()
    PARTIAL_RESPONSE = auto()
    CITATIONS = auto()
    ERROR = auto()
    FINISHED = auto()
    EXIT = auto()
    TOKEN_COUNTS = auto()

@dataclass
class PipeMessage:
    type: MessageType
    payload: Any = None

class LocalModelSignals(QObject):
    response_signal = Signal(str)
    citations_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    model_loaded_signal = Signal()
    model_unloaded_signal = Signal()
    token_count_signal = Signal(str)

class LocalModelChat:
    def __init__(self):
        self.model_process = None
        self.model_pipe = None
        self.current_model = None
        self.signals = LocalModelSignals()

    def start_model_process(self, model_name):
        if self.current_model != model_name:
            if self.is_model_loaded():
                self.terminate_current_process()

            parent_conn, child_conn = Pipe()
            self.model_pipe = parent_conn
            self.model_process = Process(target=self._local_model_process, args=(child_conn, model_name), daemon=True)
            self.model_process.start()
            self.current_model = model_name
            self._start_listening_thread()
            self.signals.model_loaded_signal.emit()
        else:
            logging.warning(f"Model {model_name} is already loaded")

    def terminate_current_process(self):
        if self.model_process is not None:
            try:
                if self.model_pipe:
                    try:
                        self.model_pipe.send(PipeMessage(MessageType.EXIT))
                    except (BrokenPipeError, OSError):
                        logging.warning("Pipe already closed")
                    finally:
                        self.model_pipe.close()
                        self.model_pipe = None
                
                process = self.model_process
                self.model_process = None
                
                if process.is_alive():
                    process.join(timeout=10)
                    if process.is_alive():
                        logging.warning("Process did not terminate, forcing termination")
                        process.terminate()
                        process.join(timeout=5)
            except Exception as e:
                logging.exception(f"Error during process termination: {e}")
        else:
            logging.warning("No process to terminate")

        self.model_pipe = None
        self.model_process = None
        self.current_model = None
        time.sleep(0.5)
        self.signals.model_unloaded_signal.emit()

    def start_chat(self, user_question, selected_model, selected_database):
        if not self.model_pipe:
            self.signals.error_signal.emit("Model not loaded. Please start a model first.")
            return

        self.model_pipe.send(PipeMessage(
            MessageType.QUESTION, 
            (user_question, selected_model, selected_database)
        ))

    def is_model_loaded(self):
        return self.model_process is not None and self.model_process.is_alive()

    def eject_model(self):
        self.terminate_current_process()

    def _start_listening_thread(self):
        import threading

        if hasattr(self, "_stop_listener_event"):
            self._stop_listener_event.set()
            if getattr(self, "listener_thread", None) and self.listener_thread.is_alive():
                self.listener_thread.join()

        self._stop_listener_event = threading.Event()
        self.listener_thread = threading.Thread(
            target=self._listen_for_response,
            args=(self._stop_listener_event,),
            daemon=True,
        )
        self.listener_thread.start()

    def _listen_for_response(self, stop_event):
        while not stop_event.is_set():
            if not self.model_pipe or not isinstance(self.model_pipe, PipeConnection):
                break
            try:
                if self.model_pipe.poll(timeout=1):
                    message = self.model_pipe.recv()
                    if message.type in [MessageType.RESPONSE, MessageType.PARTIAL_RESPONSE]:
                        self.signals.response_signal.emit(message.payload)
                    elif message.type == MessageType.CITATIONS:
                        self.signals.citations_signal.emit(message.payload)
                    elif message.type == MessageType.ERROR:
                        self.signals.error_signal.emit(message.payload)
                    elif message.type == MessageType.FINISHED:
                        self.signals.finished_signal.emit()
                        if message.payload == MessageType.EXIT:
                            break
                    elif message.type == MessageType.TOKEN_COUNTS:
                        self.signals.token_count_signal.emit(message.payload)
                else:
                    time.sleep(0.1)
            except (BrokenPipeError, EOFError, OSError):
                break
            except Exception as e:
                logging.warning(f"Unexpected error in _listen_for_response: {str(e)}")
                break
        self.cleanup_listener_resources()

    def cleanup_listener_resources(self):
        self.model_pipe = None
        self.model_process = None
        self.current_model = None

    @staticmethod
    def _local_model_process(conn, model_name):
        model_instance = module_chat.choose_model(model_name)
        query_vector_db = None
        current_database = None
        try:
            while True:
                try:
                    message = conn.recv()
                    if message.type == MessageType.QUESTION:
                        user_question, _, selected_database = message.payload
                        if query_vector_db is None or current_database != selected_database:
                            query_vector_db = QueryVectorDB.get_instance(selected_database)
                            current_database = selected_database
                        contexts, metadata_list = query_vector_db.search(user_question)
                        if not contexts:
                            conn.send(PipeMessage(MessageType.ERROR, "No relevant contexts found."))
                            conn.send(PipeMessage(MessageType.FINISHED))
                            continue
                        max_context_tokens = model_instance.max_length - 100
                        context_tokens = len(model_instance.tokenizer.encode("\n\n---\n\n".join(contexts)))

                        if context_tokens > max_context_tokens:
                            logging.warning(f"Context tokens ({context_tokens}) exceed max context limit ({max_context_tokens})")
                            error_message = (
                                "The contexts received from the vector database exceed the chat model's context limit.\n\n"
                                "You can either:\n"
                                "1) Adjust the chunk size setting when creating the database;\n"
                                "2) Adjust the search settings (e.g. relevancy, number of contexts to return, etc.);\n"
                                "3) Choose a chat model with a larger context."
                            )
                            conn.send(PipeMessage(MessageType.ERROR, error_message))
                            conn.send(PipeMessage(MessageType.FINISHED))
                            continue

                        augmented_query = f"{rag_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + user_question

                        prepend_token_count = len(model_instance.tokenizer.encode(rag_string))
                        context_token_count = len(model_instance.tokenizer.encode("\n\n---\n\n".join(contexts)))
                        user_question_token_count = len(model_instance.tokenizer.encode(user_question))

                        full_response = ""
                        buffer = ""
                        for partial_response in module_chat.generate_response(model_instance, augmented_query):
                            full_response += partial_response
                            buffer += partial_response

                            if len(buffer) >= 50 or '\n' in buffer:
                                conn.send(PipeMessage(MessageType.PARTIAL_RESPONSE, buffer))
                                buffer = ""

                        if buffer:
                            conn.send(PipeMessage(MessageType.PARTIAL_RESPONSE, buffer))

                        response_token_count = len(model_instance.tokenizer.encode(full_response))
                        remaining_tokens = model_instance.max_length - (prepend_token_count + user_question_token_count + context_token_count + response_token_count)
                        total_tokens = prepend_token_count + context_token_count + user_question_token_count + response_token_count

                        token_count_string = (
                            f"<span style='color:#2ECC40;'>available tokens ({model_instance.max_length})</span>"
                            f"<span style='color:#FF4136;'> - rag instruction ({prepend_token_count})"
                            f" - query ({user_question_token_count})"
                            f" - contexts ({context_token_count})"
                            f" - response ({response_token_count})</span>"
                            f"<span style='color:white;'> = {remaining_tokens} remaining tokens.</span>"
                        )

                        conn.send(PipeMessage(MessageType.TOKEN_COUNTS, token_count_string))

                        script_dir = PROJECT_ROOT
                        with open(script_dir / 'chat_history.txt', 'w', encoding='utf-8') as f:

                            normalized_response = full_response
                            f.write(normalized_response)
                        citations = format_citations(metadata_list)
                        conn.send(PipeMessage(MessageType.CITATIONS, citations))
                        conn.send(PipeMessage(MessageType.FINISHED))
                    elif message.type == MessageType.EXIT:
                        break
                except EOFError:
                    logging.warning("Connection closed by main process.")
                    break
                except Exception as e:
                    logging.exception(f"Error in local_model_process: {e}")
                    conn.send(PipeMessage(MessageType.ERROR, str(e)))
                    conn.send(PipeMessage(MessageType.FINISHED))
        finally:
            try:
                if hasattr(model_instance, 'cleanup'):
                    model_instance.cleanup()
            finally:
                conn.close()
                my_cprint("Local chat model removed from memory.", "red")

def is_cuda_available():
    return torch.cuda.is_available()
