import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

from db.sqlite_operations import create_metadata_db
from db.text_splitter import Document


def _manage_data(monkeypatch):
    utilities = types.ModuleType("core.utilities")
    utilities.save_config_atomically = lambda *_args, **_kwargs: None
    downloader = types.ModuleType("gui.download_model")
    downloader.is_complete_download = lambda *_args, **_kwargs: True
    config = types.ModuleType("gui.tabs_databases.create_config")
    config.model_info = lambda *_args, **_kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "core.utilities", utilities)
    monkeypatch.setitem(sys.modules, "gui.download_model", downloader)
    monkeypatch.setitem(sys.modules, "gui.tabs_databases.create_config", config)
    source = Path(__file__).resolve().parents[1] / "gui" / "tabs_databases" / "manage_data.py"
    spec = importlib.util.spec_from_file_location("manage_data_rows_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_file_rows_keep_stable_ids_and_exact_chunk_counts(tmp_path, monkeypatch):
    manage_data = _manage_data(monkeypatch)
    monkeypatch.setattr(manage_data, "PROJECT_ROOT", tmp_path)
    folder = tmp_path / "Vector_DB" / "example"
    docs = [
        Document(page_content="same", metadata={"doc_id": 1, "file_name": "a.txt", "file_path": "/a.txt", "hash": "same"}),
        Document(page_content="same", metadata={"doc_id": 2, "file_name": "copy.txt", "file_path": "/copy.txt", "hash": "same"}),
        Document(page_content="same", metadata={"doc_id": 3, "file_name": "a.txt", "file_path": "/a.txt", "hash": "same"}),
    ]
    create_metadata_db(folder, docs, [("10", "same", 1), ("20", "same", 1), ("30", "same", 2), ("40", "same", 3)])

    rows = manage_data.file_rows("example")
    assert [(row[4], row[2], row[5]) for row in rows] == [(1, 3, 2), (3, 3, 2), (2, 1, 1)]
    compact = manage_data.compact_rows(rows)
    assert [row[6] for row in compact["rows"]] == [1, 3, 2]
    assert [row[2] for row in compact["rows"]] == [3, 3, 1]
    with sqlite3.connect(folder / "metadata.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM document_metadata").fetchone()[0] == 3
