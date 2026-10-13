import os
import sqlite3
import time
from contextlib import closing

import pytest

from db.sqlite_operations import (
    METADATA_SCHEMA_VERSION,
    bump_version,
    create_metadata_db,
    database_stamp,
    stored_database_info,
    upgrade_metadata_db,
)
from db.text_splitter import Document

PAST = 1700000000.25


def _new_database(db_dir):
    document = Document(page_content="text", metadata={"doc_id": 1, "file_name": "a.txt", "hash": "ha", "file_path": "C:/a.txt"})
    create_metadata_db(db_dir, [document], [("1", "ha", 1), ("2", "ha", 1)])


def _execute(db_dir, *statements):
    with closing(sqlite3.connect(db_dir / "metadata.db")) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.commit()


def _rows(db_dir, sql):
    with closing(sqlite3.connect(db_dir / "metadata.db")) as connection:
        return connection.execute(sql).fetchall()


def _old_database(db_dir):
    db_dir.mkdir(parents=True, exist_ok=True)
    _execute(
        db_dir,
        "CREATE TABLE document_metadata (id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, hash TEXT, file_path TEXT, page_content TEXT)",
        "CREATE TABLE hash_chunk_ids (tiledb_id TEXT PRIMARY KEY, hash TEXT)",
        "INSERT INTO document_metadata (file_name, hash, file_path, page_content) VALUES ('a.txt', 'ha', 'C:/a.txt', 'x')",
        "INSERT INTO hash_chunk_ids (tiledb_id, hash) VALUES ('1', 'ha'), ('2', 'ha')",
    )
    os.utime(db_dir / "metadata.db", (PAST, PAST))


def test_new_database_records_its_creation_date_and_version(tmp_path):
    started = time.time()
    _new_database(tmp_path)
    created, version = _rows(tmp_path, "SELECT created_at, version FROM database_info")[0]
    assert started - 1 <= created <= time.time() + 1 and version == 1
    assert _rows(tmp_path, "PRAGMA user_version")[0][0] == METADATA_SCHEMA_VERSION == 4
    assert database_stamp(tmp_path) == {"created": created, "version": 1}


def test_upgrade_keeps_the_date_an_old_database_was_created(tmp_path):
    _old_database(tmp_path)
    assert database_stamp(tmp_path) == pytest.approx({"created": PAST, "version": 1})
    assert upgrade_metadata_db(tmp_path) is True
    assert (tmp_path / "metadata.db").stat().st_mtime > PAST + 1000
    assert database_stamp(tmp_path) == pytest.approx({"created": PAST, "version": 1})
    assert _rows(tmp_path, "SELECT document_id FROM hash_chunk_ids") == [(1,), (1,)]
    assert upgrade_metadata_db(tmp_path) is False


def test_upgrade_of_a_version_2_database_only_adds_the_record(tmp_path):
    _new_database(tmp_path)
    _execute(tmp_path, "DROP TABLE database_info", "PRAGMA user_version = 2")
    os.utime(tmp_path / "metadata.db", (PAST, PAST))
    assert upgrade_metadata_db(tmp_path) is True
    assert database_stamp(tmp_path) == pytest.approx({"created": PAST, "version": 1})
    assert _rows(tmp_path, "SELECT tiledb_id, document_id FROM hash_chunk_ids ORDER BY tiledb_id") == [("1", 1), ("2", 1)]


def test_version_rises_only_with_a_committed_change(tmp_path):
    _new_database(tmp_path)
    with closing(sqlite3.connect(tmp_path / "metadata.db")) as connection:
        bump_version(connection)
        connection.rollback()
        assert stored_database_info(connection)["version"] == 1
        bump_version(connection)
        connection.commit()
    assert database_stamp(tmp_path)["version"] == 2


def test_version_needs_an_upgraded_database(tmp_path):
    _old_database(tmp_path)
    with closing(sqlite3.connect(tmp_path / "metadata.db")) as connection:
        assert stored_database_info(connection) is None
        with pytest.raises(RuntimeError, match="upgrade_metadata_db"):
            bump_version(connection)


def test_reading_the_stamp_never_writes(tmp_path):
    _old_database(tmp_path)
    before = (tmp_path / "metadata.db").stat().st_mtime_ns
    database_stamp(tmp_path)
    assert (tmp_path / "metadata.db").stat().st_mtime_ns == before
