import logging
import re

import requests
from openai import OpenAI
from PySide6.QtCore import QThread

from db.database_interactions import get_query_db
from chat.base import ChatSignals, load_chat_config, save_metadata, build_augmented_query, cleanup_gpu
from core.utilities import format_citations
from core.constants import system_message, THINKING_TAGS

_ALL_THINKING_TAGS = [t for pair in THINKING_TAGS.values() for t in pair]
_START_THINKING_TAGS = frozenset(s for s, _ in THINKING_TAGS.values())
_THINKING_TAG_RE = re.compile("|".join(re.escape(t) for t in _ALL_THINKING_TAGS))


def _strip_thinking(buffer, in_thinking):
    """Process buffer, toggling in_thinking at each tag match.

    Returns (text_to_yield, new_buffer, new_in_thinking). Holds back any tail
    that could be the start of a partial tag so a tag split across chunks
    (e.g. '<thi' then 'nk>') is still detected on the next call.
    """
    out = []
    pos = 0
    cur_in = in_thinking
    while True:
        m = _THINKING_TAG_RE.search(buffer, pos)
        if m is None:
            break
        if not cur_in:
            out.append(buffer[pos:m.start()])
        cur_in = m.group(0) in _START_THINKING_TAGS
        pos = m.end()

    tail = buffer[pos:]
    hold = 0
    for t in _ALL_THINKING_TAGS:
        max_i = min(len(t) - 1, len(tail))
        for i in range(max_i, 0, -1):
            if tail.endswith(t[:i]):
                if i > hold:
                    hold = i
                break

    if hold:
        flushable = tail[:-hold]
        new_buffer = tail[-hold:]
    else:
        flushable = tail
        new_buffer = ""

    if not cur_in:
        out.append(flushable)

    return "".join(out), new_buffer, cur_in

class LMStudioChat:
    def __init__(self):
        self.signals = ChatSignals()
        self.config = load_chat_config()
        self.query_vector_db = None

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
        buffer = ""
        for chunk in stream:
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

            buffer += content
            text, buffer, in_thinking_block = _strip_thinking(buffer, in_thinking_block)
            if not text:
                continue
            if first_content:
                text = text.lstrip()
                if not text:
                    continue
                first_content = False
            yield text

        if not show_thinking and buffer and not in_thinking_block:
            tail = buffer.lstrip() if first_content else buffer
            if tail:
                yield tail

    def handle_response_and_cleanup(self, full_response, metadata_list):
        citations = format_citations(metadata_list)
        if self.query_vector_db:
            self.query_vector_db.cleanup()
        cleanup_gpu()
        return citations

    def ask_local_chatgpt(self, query, selected_database):
        if self.query_vector_db is None or self.query_vector_db.selected_database != selected_database:
            self.query_vector_db = get_query_db(selected_database)

        contexts, metadata_list = self.query_vector_db.search(query)
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
        finally:
            self.lm_studio_chat.signals.finished_signal.emit()

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
