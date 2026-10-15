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
    assert [(row[4], row[2], row[5], row[6]) for row in rows] == [(1, 2, 2, 3), (3, 1, 2, 3), (2, 1, 1, 1)]
    compact = manage_data.compact_rows(rows)
    assert [row[6] for row in compact["rows"]] == [1, 3, 2]
    assert [row[2] for row in compact["rows"]] == [2, 1, 1]
    assert [row[7] for row in compact["rows"]] == [3, 3, 1]
    with sqlite3.connect(folder / "metadata.db") as connection:
        assert connection.execute("SELECT COUNT(*) FROM document_metadata").fetchone()[0] == 3


def test_legacy_file_list_does_not_upgrade_database(tmp_path, monkeypatch):
    manage_data = _manage_data(monkeypatch)
    monkeypatch.setattr(manage_data, "PROJECT_ROOT", tmp_path)
    folder = tmp_path / "Vector_DB" / "legacy"
    folder.mkdir(parents=True)
    metadata_path = folder / "metadata.db"
    with sqlite3.connect(metadata_path) as connection:
        connection.executescript(
            "CREATE TABLE document_metadata (id INTEGER PRIMARY KEY, file_name TEXT, file_path TEXT, hash TEXT);"
            "CREATE TABLE hash_chunk_ids (tiledb_id TEXT PRIMARY KEY, hash TEXT);"
            "INSERT INTO document_metadata VALUES (1, 'a.txt', '/a.txt', 'same'), (2, 'b.txt', '/b.txt', 'same');"
            "INSERT INTO hash_chunk_ids VALUES ('10', 'same'), ('20', 'same'), ('30', 'same'), ('40', 'same');"
        )
    before = metadata_path.stat().st_mtime_ns

    rows = manage_data.file_rows("legacy")

    assert [row[2] for row in rows] == [2, 2]
    assert metadata_path.stat().st_mtime_ns == before
    with sqlite3.connect(metadata_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
