import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from db import database_mutations, stage_rebuild
from db.sqlite_operations import create_metadata_db, delete_document_metadata, removed_chunk_ids
from db.text_splitter import Document


@pytest.mark.parametrize("failure", [False, True])
def test_real_index_is_unchanged_and_sqlite_removal_is_atomic(tmp_path, failure):
    tiledb = pytest.importorskip("tiledb")
    stage_rebuild._setup_tiledb_dlls()
    vector_search = pytest.importorskip("tiledb.vector_search")

    ids = np.arange(10, 80, 10, dtype=np.uint64)
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.98, 0.02, 0.0],
         [0.97, 0.03, 0.0], [0.96, 0.04, 0.0], [0.0, 1.0, 0.0], [0.2, 0.8, 0.0]],
        dtype=np.float32,
    )
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(tiledb.Dim(name="id", domain=(0, np.iinfo(np.uint64).max - 20000), tile=10000, dtype=np.uint64)),
        attrs=[
            tiledb.Attr(name="vector", dtype=np.dtype([("", np.float32)] * 3)),
            tiledb.Attr(name="text", dtype=str, var=True),
            tiledb.Attr(name="metadata", dtype=str, var=True),
        ],
        sparse=True,
    )
    vectors_uri = str(tmp_path / "vectors")
    tiledb.Array.create(vectors_uri, schema)
    with tiledb.open(vectors_uri, mode="w") as array:
        array[ids] = {
            "vector": np.ascontiguousarray(vectors).view([("", np.float32)] * 3).reshape(-1),
            "text": np.asarray([f"chunk {value}" for value in ids], dtype=object),
            "metadata": np.asarray(["{}"] * len(ids), dtype=object),
        }

    vector_search.ingest(
        index_type="FLAT", index_uri=str(tmp_path / "vector_index"),
        input_vectors=vectors, external_ids=ids, dimensions=3,
    )
    documents = [
        Document(page_content="removed", metadata={"doc_id": 1, "file_name": "removed.txt", "file_path": "/removed.txt", "hash": "removed"}),
        Document(page_content="remaining", metadata={"doc_id": 2, "file_name": "remaining.txt", "file_path": "/remaining.txt", "hash": "remaining"}),
    ]
    mappings = [(str(int(value)), "removed" if index < 5 else "remaining", 1 if index < 5 else 2) for index, value in enumerate(ids)]
    create_metadata_db(tmp_path, documents, mappings)

    if failure:
        def fail_after_delete(connection, document_id):
            delete_document_metadata(connection, document_id)
            raise RuntimeError("simulated metadata failure")

        with patch.object(database_mutations, "delete_document_metadata", side_effect=fail_after_delete):
            with pytest.raises(RuntimeError, match="simulated metadata failure"):
                database_mutations.remove_document(tmp_path, 1)
    else:
        result = database_mutations.remove_document(tmp_path, 1)
        assert result["removed_chunks"] == 5

    with sqlite3.connect(tmp_path / "metadata.db") as connection:
        assert connection.execute("SELECT version FROM database_info").fetchone()[0] == (1 if failure else 2)
        assert connection.execute("SELECT COUNT(*) FROM document_metadata").fetchone()[0] == (2 if failure else 1)
        assert connection.execute("SELECT COUNT(*) FROM index_changes").fetchone()[0] == (0 if failure else 1)
    assert removed_chunk_ids(tmp_path) == ([] if failure else [10, 20, 30, 40, 50])

    index = vector_search.FlatIndex(uri=str(tmp_path / "vector_index"))
    _distances, found = index.query(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), k=2)
    assert set(found[0].tolist()) == {10, 20}
