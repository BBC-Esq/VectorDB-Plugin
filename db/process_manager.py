import logging
import threading
import multiprocessing
from typing import List

logger = logging.getLogger(__name__)


class ProcessManager:
    """Singleton that tracks every multiprocessing.Process spawned by the app
    and provides a graceful, time-bounded cleanup path on shutdown.

    Cleanup escalates: terminate (with timeout) -> kill (with timeout) -> close.
    Without this, an unclean GUI exit can leak Python child processes that hold
    open TileDB arrays, model weights, or CUDA contexts.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.processes: List[multiprocessing.Process] = []
        self.lock = threading.Lock()
        self._initialized = True

    def register(self, process: multiprocessing.Process):
        with self.lock:
            self.processes.append(process)
            logger.debug(f"Registered process {process.pid if process.pid else 'pending'}")

    def unregister(self, process: multiprocessing.Process):
        with self.lock:
            if process in self.processes:
                self.processes.remove(process)
                logger.debug(f"Unregistered process {process.pid if process.pid else 'unknown'}")

    def cleanup_one(self, process: multiprocessing.Process, timeout: float = 5.0) -> bool:
        if not process or not process.is_alive():
            return True

        try:
            logger.debug(f"Terminating process {process.pid}")
            process.terminate()
            process.join(timeout=timeout)

            if process.is_alive():
                logger.warning(f"Process {process.pid} did not terminate, killing")
                process.kill()
                process.join(timeout=1.0)

            if hasattr(process, 'close'):
                try:
                    process.close()
                except Exception:
                    pass

            self.unregister(process)
            return not process.is_alive()
        except Exception as e:
            logger.error(f"Error cleaning up process: {e}")
            return False

    def cleanup_all(self, timeout: float = 5.0):
        with self.lock:
            processes_copy = self.processes[:]

        for process in processes_copy:
            self.cleanup_one(process, timeout)

        with self.lock:
            remaining = len(self.processes)
            if remaining > 0:
                logger.warning(f"{remaining} processes could not be cleaned up")
            self.processes.clear()

    def get_active_count(self) -> int:
        with self.lock:
            return sum(1 for p in self.processes if p.is_alive())


_manager_instance = None


def get_process_manager() -> ProcessManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ProcessManager()
    return _manager_instance
