import logging

import requests
from PySide6.QtCore import QThread

from chat.common import ChatSignals, load_chat_config
from core.utilities import format_citations
from core.constants import system_message, THINKING_TAGS

_THINKING_PAIRS = list(THINKING_TAGS.values())
STOPPED_WHILE_THINKING = (
    "The model stopped while it was still thinking, so it gave no answer. Turn on Show thinking in the "
    "Chat Backend Settings to see its reasoning, or raise the response length limit in LM Studio."
)


class _HiddenThinking:
    def __init__(self):
        self.buffer = ""
        self.closing = None
        self.answering = False

    def feed(self, text):
        if self.answering:
            return text
        self.buffer += text
        while True:
            if self.closing is not None:
                end = self.buffer.find(self.closing)
                if end < 0:
                    self.buffer = self.buffer[-(len(self.closing) - 1):]
                    return ""
                self.buffer = self.buffer[end + len(self.closing):]
                self.closing = None
            start = self.buffer.lstrip()
            opened = next(((opening, closing) for opening, closing in _THINKING_PAIRS if start.startswith(opening)), None)
            if opened:
                self.buffer = start[len(opened[0]):]
                self.closing = opened[1]
                continue
            if any(opening.startswith(start) for opening, _ in _THINKING_PAIRS):
                return ""
            self.answering = True
            self.buffer = ""
            return start

    def finish(self):
        if self.closing is not None:
            raise RuntimeError(STOPPED_WHILE_THINKING)
        leftover, self.buffer = self.buffer.lstrip(), ""
        return leftover


class LMStudioChat:
    def __init__(self):
        self.signals = ChatSignals()
        self.config = load_chat_config()

    def connect_to_local_chatgpt(self, prompt):
        from openai import OpenAI

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

        hidden = _HiddenThinking()
        first_content = True
        for chunk in stream:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is None:
                continue
            content = chunk.choices[0].delta.content

            if show_thinking:
                if first_content:
                    content = content.lstrip()
                    if not content:
                        continue
                    first_content = False
                yield content
                continue

            text = hidden.feed(content)
            if text:
                yield text

        if not show_thinking:
            tail = hidden.finish()
            if tail:
                yield tail

    def ask_local_chatgpt(self, query, selected_database):
        from db.search_process import search_database
        from chat.base import build_augmented_query, save_metadata

        contexts, metadata_list = search_database(selected_database, query)
        save_metadata(metadata_list)

        if not contexts:
            self.signals.error_signal.emit(
                "No chunks passed the similarity threshold. "
                "Try lowering the 'Similarity' setting in the Database Query settings tab."
            )
            self.signals.finished_signal.emit()
            return

        augmented_query = build_augmented_query(contexts, query)

        full_response = ""
        response_generator = self.connect_to_local_chatgpt(augmented_query)
        for response_chunk in response_generator:
            self.signals.response_signal.emit(response_chunk)
            full_response += response_chunk

        self.signals.response_signal.emit("\n")

        citations = format_citations(metadata_list)
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
        finally:
            self.lm_studio_chat.signals.finished_signal.emit()

def is_lm_studio_available():
    try:
        response = requests.get("http://127.0.0.1:1234/v1/models/", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
