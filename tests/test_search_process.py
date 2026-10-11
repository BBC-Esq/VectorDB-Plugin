import threading
import time
from multiprocessing import Pipe

import pytest
import yaml

import core.utilities as utilities
import db.database_interactions as database_interactions
from db import search_process
from db.search_process import SearchService


class FakeQueryDB:
    def __init__(self, events, name):
        self.name = name
        events.append(("load", name))

    def search(self, question):
        if question == "broken":
            raise RuntimeError("the index is unreadable")
        return [f"{self.name}: {question}"], [{"file_path": "C:/a.txt", "similarity_score": 0.9}]


class ThreadProcess:
    def __init__(self, target, args):
        self._thread = threading.Thread(target=target, args=args, daemon=True)
        self._killed = False
        self.pid = id(self)

    def start(self):
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive() and not self._killed

    def join(self, timeout=None):
        self._thread.join(timeout)

    def terminate(self):
        self._killed = True

    kill = terminate


def thread_factory(target, processes, events=None):
    def start(name):
        if events is not None:
            events.append(("start", name, all(not p.is_alive() for p in processes)))
        parent, child = Pipe()
        process = ThreadProcess(target, (child, name))
        process.start()
        processes.append(process)
        return process, parent
    return start


@pytest.fixture
def service(monkeypatch):
    events = []
    processes = []
    settings = {"device": "cpu"}
    monkeypatch.setattr(utilities, "configure_logging", lambda level="INFO": None)
    monkeypatch.setattr(database_interactions, "QueryVectorDB", lambda name: FakeQueryDB(events, name))
    svc = SearchService(process_factory=thread_factory(search_process.serve, processes, events),
                        fingerprint=lambda name: (name, settings["device"]), timeout=5)
    yield svc, events, processes, settings
    svc.release()


def test_the_same_database_stays_loaded(service):
    svc, events, processes, _ = service
    assert svc.search("one", "first") == (["one: first"], [{"file_path": "C:/a.txt", "similarity_score": 0.9}])
    svc.search("one", "second")
    assert events == [("start", "one", True), ("load", "one")]
    assert len(processes) == 1 and processes[0].is_alive() and svc.database == "one"


def test_a_new_database_ends_the_old_process_before_loading(service):
    svc, events, processes, _ = service
    svc.search("one", "q")
    assert svc.search("two", "q")[0] == ["two: q"]
    assert events == [("start", "one", True), ("load", "one"), ("start", "two", True), ("load", "two")]
    assert not processes[0].is_alive() and processes[1].is_alive() and svc.database == "two"


def test_changed_settings_reload_the_database(service):
    svc, events, processes, settings = service
    svc.search("one", "q")
    settings["device"] = "cuda"
    svc.search("one", "q")
    assert len(processes) == 2 and not processes[0].is_alive()


def test_release_ends_the_process(service):
    svc, events, processes, _ = service
    svc.search("one", "q")
    assert svc.release("other") is False and processes[0].is_alive()
    assert svc.release("one") is True
    assert not processes[0].is_alive() and svc.database is None


def test_errors_are_reported_and_the_database_stays_loaded(service):
    svc, events, processes, _ = service
    with pytest.raises(RuntimeError, match="the index is unreadable"):
        svc.search("one", "broken")
    assert svc.search("one", "q")[0] == ["one: q"]
    assert len(processes) == 1


def test_a_process_that_died_is_replaced(service):
    svc, events, processes, _ = service
    svc.search("one", "q")
    processes[0].terminate()
    assert svc.search("one", "q")[0] == ["one: q"]
    assert len(processes) == 2


def test_a_search_that_never_answers_times_out():
    processes = []
    svc = SearchService(process_factory=thread_factory(lambda conn, name: time.sleep(1), processes),
                        fingerprint=lambda name: name, timeout=0.3)
    with pytest.raises(RuntimeError, match="took longer than 0.3 seconds"):
        svc.search("one", "q")
    assert not processes[0].is_alive() and svc.database is None


def test_release_interrupts_a_running_search():
    processes = []
    svc = SearchService(process_factory=thread_factory(lambda conn, name: time.sleep(1), processes),
                        fingerprint=lambda name: name, timeout=30)
    outcome = {}

    def run():
        try:
            svc.search("one", "q")
        except RuntimeError as e:
            outcome["error"] = str(e)

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(0.3)
    assert svc.release() is True
    worker.join(5)
    assert outcome["error"] == "The search was stopped." and svc.database is None


def test_fingerprint_follows_the_settings_and_the_database_files(tmp_path, monkeypatch):
    monkeypatch.setattr(search_process, "PROJECT_ROOT", tmp_path)

    def write_config(device):
        cfg = {"created_databases": {"one": {"model": "Models/vector/m"}}, "Compute_Device": {"database_query": device}, "database": {"half": False}}
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    write_config("cpu")
    folder = tmp_path / "Vector_DB" / "one"
    folder.mkdir(parents=True)
    (folder / "metadata.db").write_bytes(b"a")
    first = search_process.database_fingerprint("one")
    assert first == search_process.database_fingerprint("one")
    write_config("cuda")
    second = search_process.database_fingerprint("one")
    assert second != first
    (folder / "metadata.db").write_bytes(b"ab")
    assert search_process.database_fingerprint("one") != second
    (tmp_path / "config.yaml").write_text("created_databases: [broken", encoding="utf-8")
    assert search_process.database_fingerprint("one")[0] == "one"
