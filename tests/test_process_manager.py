import logging
import multiprocessing
import time

from db.process_manager import ProcessManager


def test_cleanup_stops_a_process_and_tolerates_a_closed_one(caplog):
    manager = ProcessManager()
    process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,), daemon=True)
    manager.register(process)
    process.start()
    with caplog.at_level(logging.ERROR, logger="db.process_manager"):
        assert manager.cleanup_one(process, timeout=5) is True
        assert manager.cleanup_one(process, timeout=5) is True
    assert process not in manager.processes
    assert not caplog.records
