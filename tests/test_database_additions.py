import sqlite3
import sys
import types
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from db import database_additions
from db.sqlite_operations import create_metadata_db, pending_changes, stored_database_info


class Vectors:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, positions):
        return Vectors([self.values[i] for i in positions])


def document(doc_id, path, content, digest):
    return SimpleNamespace(page_content=content, metadata={
        "doc_id": doc_id, "file_name": Path(path).name, "file_path": str(path), "hash": digest,
    })


@pytest.fixture
def database(tmp_path):
    folder = tmp_path / "database"
    folder.mkdir()
    (folder / "vectors").mkdir()
    original = tmp_path / "original.txt"
    original.write_text("old", encoding="utf-8")
    create_metadata_db(folder, [document(1, original, "old", "old-hash")], [("10", "old-hash", 1)])
    return folder, original


def fake_pipeline(monkeypatch, prepared, mappings):
    calls = []

    def prepare(name, source, first_document_id):
        calls.append(("prepare", name, first_document_id, source))
        return prepared

    def append(folder, chunks):
        calls.append(("append", folder, chunks))
        return mappings

    monkeypatch.setitem(sys.modules, "db.database_interactions", SimpleNamespace(
        prepare_documents=prepare, append_chunks=append,
    ))
    return calls


def test_add_documents_commits_rows_log_and_version(database, tmp_path, monkeypatch):
    folder, original = database
    source = tmp_path / "new.txt"
    source.write_text("new", encoding="utf-8")
    doc = document(2, source, "new", "new-hash")
    prepared = SimpleNamespace(
        documents=[doc], texts=["new chunk"], metadatas=[doc.metadata.copy()], vectors=Vectors([1]), not_added=[],
    )
    calls = fake_pipeline(monkeypatch, prepared, [("20", "new-hash", 2)])

    result = database_additions.add_documents("sample", folder, [source])

    assert result == {"added_documents": 1, "added_chunks": 1, "skipped": [], "not_added": []}
    assert calls[0][0:3] == ("prepare", "sample", 2)
    assert calls[1][0] == "append"
    with closing(sqlite3.connect(folder / "metadata.db")) as connection:
        assert connection.execute("SELECT file_path FROM document_metadata WHERE id = 2").fetchone() == (str(source),)
        assert connection.execute("SELECT tiledb_id FROM hash_chunk_ids WHERE document_id = 2").fetchone() == ("20",)
        assert pending_changes(connection) == {"documents": 1, "removed_chunks": 0, "added_chunks": 1}
        assert stored_database_info(connection)["version"] == 2


def test_duplicate_content_and_path_are_skipped(database, monkeypatch):
    folder, original = database
    doc = document(2, original, "old", "old-hash")
    prepared = SimpleNamespace(
        documents=[doc], texts=["old chunk"], metadatas=[doc.metadata.copy()], vectors=Vectors([1]), not_added=[],
    )
    calls = fake_pipeline(monkeypatch, prepared, [])

    result = database_additions.add_documents("sample", folder, [original])

    assert result["added_documents"] == 0
    assert result["skipped"] == ["original.txt"]
    assert all(call[0] != "append" for call in calls)
    with closing(sqlite3.connect(folder / "metadata.db")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM document_metadata").fetchone()[0] == 1
        assert stored_database_info(connection)["version"] == 1


def test_failure_after_append_rolls_back_sqlite(database, tmp_path, monkeypatch):
    folder, original = database
    source = tmp_path / "new.txt"
    source.write_text("new", encoding="utf-8")
    doc = document(2, source, "new", "new-hash")
    prepared = SimpleNamespace(
        documents=[doc], texts=["new chunk"], metadatas=[doc.metadata.copy()], vectors=Vectors([1]), not_added=[],
    )
    calls = fake_pipeline(monkeypatch, prepared, [("20", "new-hash", 2)])
    monkeypatch.setattr(database_additions, "record_addition", lambda connection, ids: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        database_additions.add_documents("sample", folder, [source])

    assert any(call[0] == "append" for call in calls)
    with closing(sqlite3.connect(folder / "metadata.db")) as connection:
        assert connection.execute("SELECT COUNT(*) FROM document_metadata").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM hash_chunk_ids").fetchone()[0] == 1
        assert pending_changes(connection)["documents"] == 0
        assert stored_database_info(connection)["version"] == 1


def test_staging_keeps_same_named_sources_separate(tmp_path):
    first = tmp_path / "one" / "notes.txt"
    second = tmp_path / "two" / "notes.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    staging = tmp_path / "private"
    staging.mkdir()

    staged = database_additions._stage_files([first, second], staging)

    assert len(staged) == 2
    assert len(list(staging.iterdir())) == 2
    assert set(staged.values()) == {str(first), str(second)}


def test_rejects_unsupported_input_without_modifying_database(database, tmp_path):
    folder, original = database
    unsupported = tmp_path / "program.exe"
    unsupported.write_bytes(b"do not ingest")

    with pytest.raises(ValueError, match="Unsupported"):
        database_additions._stage_files([unsupported], tmp_path)

    with closing(sqlite3.connect(folder / "metadata.db")) as connection:
        assert stored_database_info(connection)["version"] == 1


def test_copy_fallback_restores_the_original_source_path(tmp_path, monkeypatch):
    source = tmp_path / "notes.txt"
    source.write_text("content", encoding="utf-8")
    staging = tmp_path / "private"
    staging.mkdir()
    monkeypatch.setattr(Path, "symlink_to", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("no symlinks")))

    staged = database_additions._stage_files([source], staging)
    copied = staging / "notes.txt"
    doc = document(1, copied, "content", "digest")
    prepared = SimpleNamespace(documents=[doc], metadatas=[doc.metadata.copy()])
    database_additions._restore_source_paths(prepared, staged)

    assert copied.is_file() and not copied.is_symlink()
    assert doc.metadata["file_path"] == str(source)
    assert prepared.metadatas[0]["file_path"] == str(source)
