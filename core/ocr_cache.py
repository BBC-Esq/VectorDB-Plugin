import hashlib
import json
import os
from pathlib import Path

from core.constants import PROJECT_ROOT

CACHE_FOLDER = "Cache"


def cache_dir():
    return Path(PROJECT_ROOT) / CACHE_FOLDER / "ocr"


def _record_path(pdf_path):
    real = os.path.realpath(pdf_path)
    info = os.stat(real)
    key = f"{os.path.normcase(real)}|{info.st_size}|{info.st_mtime_ns}"
    return cache_dir() / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")


def load(pdf_path):
    record = {"checks": {}, "text": {}}
    try:
        with open(_record_path(pdf_path), "r", encoding="utf-8") as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return record
    if isinstance(stored, dict):
        for key in record:
            if isinstance(stored.get(key), dict):
                record[key] = stored[key]
    return record


def save(pdf_path, record):
    try:
        target = _record_path(pdf_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(record, f)
        temporary.replace(target)
    except OSError:
        pass
