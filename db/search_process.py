import logging
import multiprocessing
import threading
import time

import yaml

from core.constants import PROJECT_ROOT
from db.process_manager import get_process_manager

SEARCH_TIMEOUT = 600
STOPPED_UNEXPECTEDLY = "The database search stopped unexpectedly. Check the command prompt window for details."


def serve(conn, database_name):
    from core.utilities import configure_logging

    configure_logging("INFO")
    query_db = None
    while True:
        try:
            question = conn.recv()
        except (EOFError, OSError):
            break
        if question is None:
            break
        try:
            if query_db is None:
                from db.database_interactions import QueryVectorDB

                query_db = QueryVectorDB(database_name)
            contexts, metadata_list = query_db.search(question)
            conn.send({"contexts": contexts, "metadata": metadata_list})
        except Exception as e:
            logging.exception("Searching %s failed", database_name)
            conn.send({"error": f"The database could not be searched: {e}"})


def spawn_search_process(database_name):
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    process = ctx.Process(target=serve, args=(child_conn, database_name), daemon=True)
    get_process_manager().register(process)
    process.start()
    child_conn.close()
    return process, parent_conn


def database_fingerprint(database_name):
    try:
        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        cfg = None
    cfg = cfg if isinstance(cfg, dict) else {}
    entries = cfg.get("created_databases")
    entry = entries.get(database_name) if isinstance(entries, dict) else None
    model = entry.get("model") if isinstance(entry, dict) else None
    compute = cfg.get("Compute_Device")
    device = compute.get("database_query") if isinstance(compute, dict) else None
    settings = cfg.get("database")
    half = settings.get("half") if isinstance(settings, dict) else None
    folder = PROJECT_ROOT / "Vector_DB" / database_name
    stamps = []
    for name in ("metadata.db", "index_metadata.json"):
        try:
            st = (folder / name).stat()
            stamps.append((st.st_mtime_ns, st.st_size))
        except OSError:
            stamps.append(None)
    return (database_name, str(model), str(device), bool(half), tuple(stamps))


class SearchService:
    def __init__(self, process_factory=spawn_search_process, fingerprint=database_fingerprint, timeout=SEARCH_TIMEOUT):
        self._process_factory = process_factory
        self._fingerprint = fingerprint
        self._timeout = timeout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._process = None
        self._conn = None
        self._key = None
        self._database = None

    @property
    def database(self):
        process = self._process
        return self._database if process is not None and process.is_alive() else None

    @property
    def pid(self):
        process = self._process
        return process.pid if process is not None and process.is_alive() else None

    def search(self, database_name, question):
        key = self._fingerprint(database_name)
        with self._lock:
            if key != self._key or self._process is None or not self._process.is_alive():
                self._shutdown(graceful=True)
                self._process, self._conn = self._process_factory(database_name)
                self._key = key
                self._database = database_name
            result = self._exchange(question)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result["contexts"], result["metadata"]

    def _exchange(self, question):
        conn, process = self._conn, self._process
        try:
            conn.send(question)
        except (BrokenPipeError, OSError):
            self._shutdown(graceful=False)
            raise RuntimeError(STOPPED_UNEXPECTEDLY)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                if conn.poll(0.25):
                    return conn.recv()
            except (EOFError, OSError):
                self._shutdown(graceful=False)
                raise RuntimeError(STOPPED_UNEXPECTEDLY)
            if self._stop.is_set():
                self._shutdown(graceful=False)
                raise RuntimeError("The search was stopped.")
            if not process.is_alive():
                self._shutdown(graceful=False)
                raise RuntimeError(STOPPED_UNEXPECTEDLY)
            if time.monotonic() >= deadline:
                self._shutdown(graceful=False)
                raise RuntimeError(f"The database search took longer than {self._timeout} seconds and was stopped.")

    def release(self, database_name=None):
        if database_name is not None and self._database != database_name:
            return False
        self._stop.set()
        try:
            with self._lock:
                if database_name is not None and self._database != database_name:
                    return False
                self._shutdown(graceful=True)
        finally:
            self._stop.clear()
        return True

    def _shutdown(self, graceful):
        process, conn = self._process, self._conn
        self._process = self._conn = self._key = self._database = None
        if conn is not None and graceful:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        if process is not None:
            if graceful:
                process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            get_process_manager().unregister(process)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass


_service = None
_service_lock = threading.Lock()


def search_service():
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = SearchService()
    return _service


def search_database(database_name, question):
    return search_service().search(database_name, question)


def loaded_database():
    return search_service().database


def release_database(database_name=None):
    return search_service().release(database_name)
