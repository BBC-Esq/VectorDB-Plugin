import threading
from multiprocessing import Pipe

import pytest
from PySide6.QtCore import Qt

import chat.base as chat_base
from chat.local_model import LocalModelChat, MessageType, PipeMessage


class FakeTokenizer:
    def __call__(self, text):
        return {"input_ids": text.split()}

    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeModel:
    tokenizer = FakeTokenizer()
    max_length = 100000

    def create_prompt(self, query):
        return query


@pytest.fixture
def model_process(monkeypatch):
    prompts = []

    def generate(model, query):
        prompts.append(query)
        return iter(["An answer."])

    monkeypatch.setattr(chat_base, "choose_model", lambda name: FakeModel())
    monkeypatch.setattr(chat_base, "generate_response", generate)
    parent, child = Pipe()
    worker = threading.Thread(target=LocalModelChat._local_model_process, args=(child, "fake"), daemon=True)
    worker.start()

    def ask(question, contexts, metadata_list):
        parent.send(PipeMessage(MessageType.QUESTION, (question, contexts, metadata_list)))
        messages = []
        while parent.poll(30):
            message = parent.recv()
            messages.append(message)
            if message.type == MessageType.FINISHED:
                return messages
        raise AssertionError("The model process did not finish answering.")

    yield ask, prompts
    parent.send(PipeMessage(MessageType.EXIT))
    worker.join(timeout=10)


def test_answers_from_the_passages_it_is_given(model_process):
    ask, prompts = model_process
    messages = ask("What is the notice period?", ["Thirty days notice is required."],
                   [{"file_name": "lease.txt", "file_path": "C:/lease.txt", "similarity_score": 0.9}])
    kinds = [m.type for m in messages]
    assert MessageType.ERROR not in kinds
    assert "".join(m.payload for m in messages if m.type == MessageType.PARTIAL_RESPONSE) == "An answer."
    assert "Thirty days notice is required." in prompts[0] and prompts[0].endswith("What is the notice period?")
    assert "lease.txt" in next(m.payload for m in messages if m.type == MessageType.CITATIONS)


def test_each_question_uses_its_own_passages(model_process):
    ask, prompts = model_process
    ask("First?", ["Passage one."], [{"file_path": "C:/one.txt", "similarity_score": 0.8}])
    ask("Second?", ["Passage two."], [{"file_path": "C:/two.txt", "similarity_score": 0.7}])
    assert "Passage one." in prompts[0] and "Passage two." in prompts[1] and "Passage one." not in prompts[1]


def listen_until_the_pipe_closes(awaiting_reply, announce_finish):
    manager = LocalModelChat()
    events = []
    manager.signals.error_signal.connect(lambda message: events.append(("error", message)), Qt.DirectConnection)
    manager.signals.finished_signal.connect(lambda: events.append(("finished", None)), Qt.DirectConnection)
    parent, child = Pipe()
    manager.model_pipe = parent
    manager._awaiting_reply = awaiting_reply
    manager._start_listening_thread()
    if announce_finish:
        child.send(PipeMessage(MessageType.ERROR, "Failed to load model"))
        child.send(PipeMessage(MessageType.FINISHED))
    child.close()
    manager.listener_thread.join(timeout=10)
    return events


def test_a_model_process_that_dies_while_answering_is_reported():
    events = listen_until_the_pipe_closes(awaiting_reply=True, announce_finish=False)
    assert [kind for kind, _ in events] == ["error", "finished"]
    assert "stopped unexpectedly" in events[0][1]


def test_a_model_process_that_ends_after_its_reply_adds_no_error():
    events = listen_until_the_pipe_closes(awaiting_reply=True, announce_finish=True)
    assert [kind for kind, _ in events].count("error") == 1
    assert events[0] == ("error", "Failed to load model")
