import threading
from multiprocessing import Pipe

import pytest

import chat.base as chat_base
import db.database_interactions as database_interactions
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


class FakeQueryDB:
    def __init__(self, events, name):
        self.events = events
        self.selected_database = name
        events.append(("load", name))

    def search(self, query):
        return [f"A passage from {self.selected_database}."], [{"file_name": "a.txt", "file_path": "C:/a.txt", "similarity_score": 0.9}]

    def close(self):
        self.events.append(("release", self.selected_database))


@pytest.fixture
def model_process(monkeypatch):
    events = []
    monkeypatch.setattr(chat_base, "choose_model", lambda name: FakeModel())
    monkeypatch.setattr(chat_base, "generate_response", lambda model, query: iter(["An answer."]))
    monkeypatch.setattr(database_interactions, "QueryVectorDB", lambda name: FakeQueryDB(events, name))
    parent, child = Pipe()
    worker = threading.Thread(target=LocalModelChat._local_model_process, args=(child, "fake"), daemon=True)
    worker.start()

    def ask(database):
        parent.send(PipeMessage(MessageType.QUESTION, ("What does it say?", "fake", database)))
        kinds = []
        while parent.poll(30):
            message = parent.recv()
            kinds.append(message.type)
            if message.type == MessageType.FINISHED:
                return kinds
        raise AssertionError("The model process did not finish answering.")

    yield ask, events
    parent.send(PipeMessage(MessageType.EXIT))
    worker.join(timeout=10)


def test_follow_up_questions_reuse_the_loaded_database(model_process):
    ask, events = model_process
    assert MessageType.CITATIONS in ask("first")
    assert MessageType.ERROR not in ask("first")
    assert events == [("load", "first")]


def test_switching_databases_releases_the_previous_one(model_process):
    ask, events = model_process
    ask("first")
    ask("second")
    assert events == [("load", "first"), ("release", "first"), ("load", "second")]
    ask("first")
    assert events[3:] == [("release", "second"), ("load", "first")]
