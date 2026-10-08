import time
import logging
import threading
from enum import Enum, auto
from typing import Any, Optional
from dataclasses import dataclass

import torch
from multiprocessing import Process, Pipe
from PySide6.QtCore import QObject, Signal

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
        self._stop_listener()
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

    def _stop_listener(self):
        stop_event = getattr(self, "_stop_listener_event", None)
        if stop_event is not None:
            stop_event.set()
        listener = getattr(self, "listener_thread", None)
        if listener is not None and listener.is_alive() and listener is not threading.current_thread():
            listener.join(timeout=5)

    def _start_listening_thread(self):
        self._stop_listener()
        self._stop_listener_event = threading.Event()
        self.listener_thread = threading.Thread(
            target=self._listen_for_response,
            args=(self._stop_listener_event, self.model_pipe),
            daemon=True,
        )
        self.listener_thread.start()

    def _listen_for_response(self, stop_event, pipe):
        while not stop_event.is_set():
            try:
                if pipe.poll(timeout=1):
                    message = pipe.recv()
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
                if not stop_event.is_set():
                    self.signals.finished_signal.emit()
                break
            except Exception as e:
                logging.warning(f"Unexpected error in _listen_for_response: {str(e)}")
                if not stop_event.is_set():
                    self.signals.finished_signal.emit()
                break
        if not stop_event.is_set():
            self.cleanup_listener_resources(pipe)

    def cleanup_listener_resources(self, pipe=None):
        if pipe is not None and self.model_pipe is not pipe:
            return
        self.model_pipe = None
        self.model_process = None
        self.current_model = None

    @staticmethod
    def _local_model_process(conn, model_name):
        import chat.base as module_chat
        from db.database_interactions import get_query_db
        from core.utilities import format_citations, my_cprint

        try:
            model_instance = module_chat.choose_model(model_name)
        except Exception as e:
            logging.exception(f"Failed to load local model '{model_name}': {e}")
            try:
                conn.send(PipeMessage(MessageType.ERROR, f"Failed to load model '{model_name}': {e}"))
                conn.send(PipeMessage(MessageType.FINISHED))
            except (BrokenPipeError, OSError):
                pass
            conn.close()
            return
        query_vector_db = None
        current_database = None
        try:
            while True:
                try:
                    message = conn.recv()
                    if message.type == MessageType.QUESTION:
                        user_question, _, selected_database = message.payload
                        if query_vector_db is None or current_database != selected_database:
                            query_vector_db = get_query_db(selected_database)
                            current_database = selected_database
                        contexts, metadata_list = query_vector_db.search(user_question)
                        if not contexts:
                            conn.send(PipeMessage(
                                MessageType.ERROR,
                                "No chunks passed the similarity threshold. "
                                "Try lowering the 'Similarity' setting in the Database Query settings tab."
                            ))
                            conn.send(PipeMessage(MessageType.FINISHED))
                            continue
                        joined_contexts = "\n\n---\n\n".join(contexts)
                        augmented_query = f"{rag_string}\n\n---\n\n" + joined_contexts + "\n\n-----\n\n" + user_question
                        tokenizer = model_instance.tokenizer
                        prompt_token_count = len(tokenizer(model_instance.create_prompt(augmented_query))["input_ids"])

                        if prompt_token_count > model_instance.max_length:
                            logging.warning(f"Prompt tokens ({prompt_token_count}) exceed max context limit ({model_instance.max_length})")
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

                        context_token_count = len(tokenizer.encode(joined_contexts, add_special_tokens=False))
                        user_question_token_count = len(tokenizer.encode(user_question, add_special_tokens=False))
                        prepend_token_count = prompt_token_count - context_token_count - user_question_token_count

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

                        response_token_count = len(tokenizer.encode(full_response, add_special_tokens=False))
                        remaining_tokens = model_instance.max_length - (prompt_token_count + response_token_count)

                        token_count_string = (
                            f"<span style='color:#2ECC40;'>available tokens ({model_instance.max_length})</span>"
                            f"<span style='color:#FF4136;'> - rag instruction ({prepend_token_count})"
                            f" - query ({user_question_token_count})"
                            f" - contexts ({context_token_count})"
                            f" - response ({response_token_count})</span>"
                            f"<span style='color:white;'> = {remaining_tokens} remaining tokens.</span>"
                        )

                        conn.send(PipeMessage(MessageType.TOKEN_COUNTS, token_count_string))

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
