import os
import sqlite3
from contextlib import closing

import pytest

from db.sqlite_operations import (
    added_chunk_ids,
    bump_version,
    clear_changes,
    create_metadata_db,
    database_stamp,
    pending_changes,
    record_addition,
    record_removal,
    removed_chunk_ids,
    upgrade_metadata_db,
)
from db.text_splitter import Document

PAST = 1700000000.25
NO_CHANGES = {"documents": 0, "removed_chunks": 0, "added_chunks": 0}
CHANGE_TABLES = {"removed_chunks", "added_chunks", "index_changes"}


def _new_database(db_dir):
    documents = [
        Document(page_content="a", metadata={"doc_id": 1, "file_name": "a.txt", "hash": "ha", "file_path": "C:/a.txt"}),
        Document(page_content="b", metadata={"doc_id": 2, "file_name": "b.txt", "hash": "hb", "file_path": "C:/b.txt"}),
    ]
    mappings = [("11", "ha", 1), ("12", "ha", 1), ("13", "ha", 1), ("21", "hb", 2), ("22", "hb", 2)]
    create_metadata_db(db_dir, documents, mappings)


def _execute(db_dir, *statements):
    with closing(sqlite3.connect(db_dir / "metadata.db")) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.commit()


def _tables(db_dir):
    with closing(sqlite3.connect(db_dir / "metadata.db")) as connection:
        return {name for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _changes(db_dir):
    with closing(sqlite3.connect(db_dir / "metadata.db")) as connection:
        return pending_changes(connection)


def _remove(connection, document_id):
    record_removal(connection, [document_id])
    connection.execute("DELETE FROM hash_chunk_ids WHERE document_id = ?", (document_id,))
    connection.execute("DELETE FROM document_metadata WHERE id = ?", (document_id,))
    bump_version(connection)


def _add(connection):
    connection.execute("INSERT INTO document_metadata (id, file_name, hash, file_path, page_content) "
                       "VALUES (3, 'c.txt', 'hc', 'C:/c.txt', 'c')")
    connection.executemany("INSERT INTO hash_chunk_ids (tiledb_id, hash, document_id) VALUES (?, 'hc', 3)", [("31",), ("32",)])
    record_addition(connection, [3])
    bump_version(connection)


def test_new_and_upgraded_databases_have_the_change_log(tmp_path):
    _new_database(tmp_path / "new")
    assert CHANGE_TABLES <= _tables(tmp_path / "new")
    assert _changes(tmp_path / "new") == NO_CHANGES
    for name, statements in (("version3", ("DROP TABLE removed_chunks", "DROP TABLE added_chunks", "DROP TABLE index_changes",
                                           "PRAGMA user_version = 3")),
                             ("version4", ("DROP TABLE added_chunks", "PRAGMA user_version = 4"))):
        older = tmp_path / name
        _new_database(older)
        _execute(older, *statements)
        os.utime(older / "metadata.db", (PAST, PAST))
        assert upgrade_metadata_db(older) is True
        assert CHANGE_TABLES <= _tables(older)
        assert database_stamp(older)["version"] == 1


def test_removal_is_recorded_only_when_committed(tmp_path):
    _new_database(tmp_path)
    with closing(sqlite3.connect(tmp_path / "metadata.db")) as connection:
        _remove(connection, 1)
        connection.rollback()
    assert _changes(tmp_path) == NO_CHANGES
    assert removed_chunk_ids(tmp_path) == []
    assert database_stamp(tmp_path)["version"] == 1
    with closing(sqlite3.connect(tmp_path / "metadata.db")) as connection:
        _remove(connection, 1)
        connection.commit()
        logged = connection.execute("SELECT change, file_name, chunks FROM index_changes").fetchall()
    assert _changes(tmp_path) == {"documents": 1, "removed_chunks": 3, "added_chunks": 0}
    assert sorted(removed_chunk_ids(tmp_path)) == [11, 12, 13]
    assert logged == [("removed", "a.txt", 3)]
    assert database_stamp(tmp_path)["version"] == 2


def test_addition_is_recorded_only_when_committed(tmp_path):
    _new_database(tmp_path)
    with closing(sqlite3.connect(tmp_path / "metadata.db")) as connection:
        _add(connection)
        connection.rollback()
    assert _changes(tmp_path) == NO_CHANGES
    assert added_chunk_ids(tmp_path) == []
    with closing(sqlite3.connect(tmp_path / "metadata.db")) as connection:
        _add(connection)
        connection.commit()
        logged = connection.execute("SELECT change, file_name, chunks FROM index_changes").fetchall()
    assert _changes(tmp_path) == {"documents": 1, "removed_chunks": 0, "added_chunks": 2}
    assert sorted(added_chunk_ids(tmp_path)) == [31, 32]
    assert logged == [("added", "c.txt", 2)]
    assert database_stamp(tmp_path)["version"] == 2


def test_recording_needs_an_upgraded_database(tmp_path):
    _new_database(tmp_path)
    _execute(tmp_path, "DROP TABLE removed_chunks", "DROP TABLE added_chunks", "DROP TABLE index_changes",
             "PRAGMA user_version = 3")
    with closing(sqlite3.connect(tmp_path / "metadata.db")) as connection:
        with pytest.raises(RuntimeError, match="upgrade_metadata_db"):
            record_removal(connection, [1])
        with pytest.raises(RuntimeError, match="upgrade_metadata_db"):
            record_addition(connection, [1])
    assert _changes(tmp_path) == NO_CHANGES


def test_reading_changed_chunks_never_writes(tmp_path):
    _new_database(tmp_path)
    _execute(tmp_path, "DROP TABLE removed_chunks", "DROP TABLE added_chunks", "DROP TABLE index_changes",
             "PRAGMA user_version = 3")
    before = (tmp_path / "metadata.db").stat().st_mtime_ns
    assert removed_chunk_ids(tmp_path) == []
    assert added_chunk_ids(tmp_path) == []
    assert (tmp_path / "metadata.db").stat().st_mtime_ns == before


def test_clearing_the_change_log(tmp_path):
    _new_database(tmp_path)
    with closing(sqlite3.connect(tmp_path / "metadata.db")) as connection:
        _remove(connection, 2)
        _add(connection)
        connection.commit()
        clear_changes(connection)
        connection.commit()
    assert _changes(tmp_path) == NO_CHANGES
    assert removed_chunk_ids(tmp_path) == []
    assert added_chunk_ids(tmp_path) == []
