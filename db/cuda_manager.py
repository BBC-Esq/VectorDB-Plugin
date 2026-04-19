import threading
import logging
import torch
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CUDAManager:
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
        self.active_operations = 0
        self.operation_lock = threading.Lock()
        self._initialized = True

    @contextmanager
    def cuda_operation(self):
        with self.operation_lock:
            self.active_operations += 1
        try:
            yield
        finally:
            with self.operation_lock:
                self.active_operations -= 1

    def safe_empty_cache(self):
        if not torch.cuda.is_available():
            return

        with self.operation_lock:
            if self.active_operations > 0:
                logger.debug(f"Skipping cache clear: {self.active_operations} active operations")
                return

        try:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared successfully")
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")

    def force_empty_cache(self):
        if not torch.cuda.is_available():
            return

        try:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache forcibly cleared")
        except Exception as e:
            logger.warning(f"Failed to force clear CUDA cache: {e}")


_cuda_manager_instance = None


def get_cuda_manager() -> CUDAManager:
    global _cuda_manager_instance
    if _cuda_manager_instance is None:
        _cuda_manager_instance = CUDAManager()
    return _cuda_manager_instance
