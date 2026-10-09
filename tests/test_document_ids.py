import json
import sqlite3

import numpy as np
import pytest

from db import sqlite_operations
from db.sqlite_operations import (
    METADATA_SCHEMA_VERSION,
    chunk_ids_for_documents,
    create_metadata_db,
    document_group,
    documents_with_content,
    ensure_document_ids,
)
from db.text_splitter import Document

BIG = 2**64 - 30000


def _doc(doc_id, name, content_hash, path):
    metadata = {"file_name": name, "hash": content_hash, "file_path": path}
    if doc_id is not None:
        metadata["doc_id"] = doc_id
    return Document(page_content=f"text of {name}", metadata=metadata)


def _rows(db_dir, sql, params=()):
    with sqlite3.connect(db_dir / "metadata.db") as connection:
        return connection.execute(sql, params).fetchall()


def _indexes(db_dir):
    return {name for (name,) in _rows(db_dir, "SELECT name FROM sqlite_master WHERE type = 'index'")}


def _write_vectors(db_dir, chunks):
    import tiledb

    uri = str(db_dir / "vectors")
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(tiledb.Dim(name="id", domain=(0, np.iinfo(np.uint64).max - 20000), tile=10000, dtype=np.uint64)),
        attrs=[tiledb.Attr(name="text", dtype=str, var=True), tiledb.Attr(name="metadata", dtype=str, var=True)],
        sparse=True,
    )
    tiledb.Array.create(uri, schema)
    ids = np.array([int(chunk_id) for chunk_id, _ in chunks], dtype=np.uint64)
    with tiledb.open(uri, mode="w") as array:
        array[ids] = {
            "text": np.array([f"chunk {chunk_id}" for chunk_id, _ in chunks], dtype=object),
            "metadata": np.array([json.dumps(metadata) for _, metadata in chunks], dtype=object),
        }


def _old_database(db_dir, documents, mappings):
    db_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_dir / "metadata.db") as connection:
        connection.execute(
            "CREATE TABLE document_metadata (id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT, "
            "hash TEXT, file_path TEXT, page_content TEXT)"
        )
        connection.execute("CREATE TABLE hash_chunk_ids (tiledb_id TEXT PRIMARY KEY, hash TEXT)")
        connection.executemany(
            "INSERT INTO document_metadata (file_name, hash, file_path, page_content) VALUES (?, ?, ?, ?)",
            documents,
        )
        connection.executemany("INSERT INTO hash_chunk_ids (tiledb_id, hash) VALUES (?, ?)", mappings)


def test_new_database_records_document_ids(tmp_path):
    docs = [
        _doc(1, "a.txt", "ha", "C:/m/a.txt"),
        _doc(2, "contract.txt", "hc", "C:/matterA/contract.txt"),
        _doc(3, "contract.txt", "hc", "C:/matterB/contract.txt"),
        _doc(4, "memo.txt", "hm", "C:/m/memo.txt"),
        _doc(5, "memo.txt", "hm", "C:/m/memo.txt"),
    ]
    mappings = [("11", "ha", 1), ("12", "ha", 1), ("21", "hc", 2), ("31", "hc", 3), ("32", "hc", 3),
                ("41", "hm", 4), ("51", "hm", 5)]
    create_metadata_db(tmp_path, docs, mappings)

    assert _rows(tmp_path, "PRAGMA user_version")[0][0] == METADATA_SCHEMA_VERSION
    assert [row[0] for row in _rows(tmp_path, "SELECT id FROM document_metadata ORDER BY id")] == [1, 2, 3, 4, 5]
    assert dict(_rows(tmp_path, "SELECT tiledb_id, document_id FROM hash_chunk_ids")) == {
        "11": 1, "12": 1, "21": 2, "31": 3, "32": 3, "41": 4, "51": 5,
    }
    assert {"hash_chunk_ids_document", "document_metadata_hash"} <= _indexes(tmp_path)

    with sqlite3.connect(tmp_path / "metadata.db") as connection:
        assert sorted(chunk_ids_for_documents(connection, [3])) == ["31", "32"]
        assert documents_with_content(connection, "hc") == [2, 3]
        assert sorted(chunk_ids_for_documents(connection, documents_with_content(connection, "hc"))) == ["21", "31", "32"]
        assert document_group(connection, 2) == [2]
        assert document_group(connection, 4) == [4, 5]
        assert sorted(chunk_ids_for_documents(connection, document_group(connection, 5))) == ["41", "51"]
        assert document_group(connection, 99) == []
    assert ensure_document_ids(tmp_path) is False


def test_legacy_two_item_mappings_are_filled_by_content(tmp_path):
    docs = [_doc(None, "a.txt", "ha", "C:/a.txt"), _doc(None, "b.txt", "hb", "C:/b.txt")]
    create_metadata_db(tmp_path, docs, [("1", "ha"), ("2", "ha"), ("3", "hb")])

    ids = dict(_rows(tmp_path, "SELECT hash, id FROM document_metadata"))
    assert dict(_rows(tmp_path, "SELECT tiledb_id, document_id FROM hash_chunk_ids")) == {
        "1": ids["ha"], "2": ids["ha"], "3": ids["hb"],
    }
    assert _rows(tmp_path, "PRAGMA user_version")[0][0] == METADATA_SCHEMA_VERSION


def test_inconsistent_document_ids_fall_back_to_content(tmp_path):
    docs = [_doc(7, "a.txt", "ha", "C:/a.txt"), _doc(7, "b.txt", "hb", "C:/b.txt")]
    create_metadata_db(tmp_path, docs, [("1", "ha", 7), ("2", "hb", 7)])

    ids = dict(_rows(tmp_path, "SELECT hash, id FROM document_metadata"))
    assert len(set(ids.values())) == 2
    assert dict(_rows(tmp_path, "SELECT tiledb_id, document_id FROM hash_chunk_ids")) == {"1": ids["ha"], "2": ids["hb"]}


def test_upgrade_of_old_database(tmp_path):
    db_dir = tmp_path / "old_db"
    documents = [
        ("a.txt", "ha", "C:/m/a.txt", "x"),
        ("contract.txt", "hc", "C:/matterA/contract.txt", "x"),
        ("contract.txt", "hc", "C:/matterB/contract.txt", "x"),
        ("memo.txt", "hm", "C:/m/memo.txt", "x"),
        ("memo.txt", "hm", "C:/m/memo.txt", "x"),
    ]
    chunk_owner = {
        str(BIG + 1): ("ha", "C:/m/a.txt"),
        str(BIG + 2): ("ha", "C:/m/a.txt"),
        str(BIG + 3): ("hc", "C:/matterA/contract.txt"),
        str(BIG + 4): ("hc", "C:/matterB/contract.txt"),
        str(BIG + 5): ("hc", "C:/matterB/contract.txt"),
        str(BIG + 6): ("hm", "C:/m/memo.txt"),
        str(BIG + 7): ("hm", "C:/m/memo.txt"),
        "12345": ("horphan", "C:/gone.txt"),
    }
    _old_database(db_dir, documents, [(chunk_id, owner[0]) for chunk_id, owner in chunk_owner.items()])
    _write_vectors(db_dir, [(chunk_id, {"hash": h, "file_path": p}) for chunk_id, (h, p) in chunk_owner.items()])

    assert ensure_document_ids(db_dir) is True

    assigned = dict(_rows(db_dir, "SELECT tiledb_id, document_id FROM hash_chunk_ids"))
    assert assigned == {
        str(BIG + 1): 1, str(BIG + 2): 1,
        str(BIG + 3): 2,
        str(BIG + 4): 3, str(BIG + 5): 3,
        str(BIG + 6): 4, str(BIG + 7): 4,
        "12345": None,
    }
    assert _rows(db_dir, "PRAGMA user_version")[0][0] == METADATA_SCHEMA_VERSION
    assert {"hash_chunk_ids_document", "document_metadata_hash"} <= _indexes(db_dir)
    with sqlite3.connect(db_dir / "metadata.db") as connection:
        assert sorted(chunk_ids_for_documents(connection, document_group(connection, 5))) == [str(BIG + 6), str(BIG + 7)]
    assert ensure_document_ids(db_dir) is False


def test_upgrade_without_chunk_store_leaves_shared_content_unassigned(tmp_path):
    db_dir = tmp_path / "old_db"
    _old_database(
        db_dir,
        [("a.txt", "ha", "C:/a.txt", "x"), ("c.txt", "hc", "C:/A/c.txt", "x"), ("c.txt", "hc", "C:/B/c.txt", "x")],
        [("1", "ha"), ("2", "hc"), ("3", "hc")],
    )
    assert ensure_document_ids(db_dir) is True
    assert dict(_rows(db_dir, "SELECT tiledb_id, document_id FROM hash_chunk_ids")) == {"1": 1, "2": None, "3": None}


def test_failed_upgrade_changes_nothing(tmp_path, monkeypatch):
    db_dir = tmp_path / "old_db"
    _old_database(db_dir, [("a.txt", "ha", "C:/a.txt", "x")], [("1", "ha")])
    before = (db_dir / "metadata.db").read_bytes()

    real_assign = sqlite_operations._assign_document_ids

    def assign_then_fail(connection, vectors_uri):
        real_assign(connection, vectors_uri)
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(sqlite_operations, "_assign_document_ids", assign_then_fail)
    with pytest.raises(RuntimeError):
        ensure_document_ids(db_dir)

    columns = [row[1] for row in _rows(db_dir, "PRAGMA table_info(hash_chunk_ids)")]
    assert columns == ["tiledb_id", "hash"]
    assert _rows(db_dir, "PRAGMA user_version")[0][0] == 0
    assert not {"hash_chunk_ids_document", "document_metadata_hash"} & _indexes(db_dir)
    assert (db_dir / "metadata.db").read_bytes() == before

    monkeypatch.setattr(sqlite_operations, "_assign_document_ids", real_assign)
    assert ensure_document_ids(db_dir) is True
    assert dict(_rows(db_dir, "SELECT tiledb_id, document_id FROM hash_chunk_ids")) == {"1": 1}


def test_lookup_batches_large_requests(tmp_path):
    docs = [_doc(i, f"f{i}.txt", f"h{i}", f"C:/f{i}.txt") for i in range(1, 1201)]
    create_metadata_db(tmp_path, docs, [(str(i), f"h{i}", i) for i in range(1, 1201)])
    with sqlite3.connect(tmp_path / "metadata.db") as connection:
        assert len(chunk_ids_for_documents(connection, range(1, 1201))) == 1200
