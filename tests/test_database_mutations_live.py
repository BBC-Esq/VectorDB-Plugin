import json
from unittest.mock import patch

import numpy as np
import pytest

from db import database_mutations
from db.sqlite_operations import create_metadata_db, delete_document_metadata
from db.text_splitter import Document


@pytest.mark.parametrize("fail_after_delete", [False, True])
def test_real_index_returns_results_after_removal_or_rollback(tmp_path, fail_after_delete):
    tiledb = pytest.importorskip("tiledb")
    vector_search = pytest.importorskip("tiledb.vector_search")

    ids = np.arange(10, 80, 10, dtype=np.uint64)
    vectors = np.asarray(
        [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.98, 0.02, 0.0], [0.97, 0.03, 0.0], [0.96, 0.04, 0.0], [0.0, 1.0, 0.0], [0.2, 0.8, 0.0]],
        dtype=np.float32,
    )
    vectors_uri = str(tmp_path / "vectors")
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(tiledb.Dim(name="id", domain=(0, np.iinfo(np.uint64).max - 20000), tile=10000, dtype=np.uint64)),
        attrs=[
            tiledb.Attr(name="vector", dtype=np.dtype([("", np.float32)] * 3)),
            tiledb.Attr(name="text", dtype=str, var=True),
            tiledb.Attr(name="metadata", dtype=str, var=True),
        ],
        sparse=True,
    )
    tiledb.Array.create(vectors_uri, schema)
    with tiledb.open(vectors_uri, mode="w") as array:
        array[ids] = {
            "vector": np.ascontiguousarray(vectors).view([("", np.float32)] * 3).reshape(-1),
            "text": np.asarray([f"chunk {value}" for value in ids], dtype=object),
            "metadata": np.asarray(["{}"] * len(ids), dtype=object),
        }

    vector_search.ingest(
        index_type="FLAT",
        index_uri=str(tmp_path / "vector_index"),
        input_vectors=vectors,
        external_ids=ids,
        dimensions=3,
    )
    documents = [
        Document(page_content="removed", metadata={"doc_id": 1, "file_name": "removed.txt", "file_path": "/removed.txt", "hash": "removed"}),
        Document(page_content="remaining", metadata={"doc_id": 2, "file_name": "remaining.txt", "file_path": "/remaining.txt", "hash": "remaining"}),
    ]
    mappings = [(str(int(value)), "removed" if index < 5 else "remaining", 1 if index < 5 else 2) for index, value in enumerate(ids)]
    create_metadata_db(tmp_path, documents, mappings)
    (tmp_path / "index_metadata.json").write_text(json.dumps({"index_type": "FLAT", "num_vectors": len(ids)}), encoding="utf-8")

    if fail_after_delete:
        def fail_metadata(connection, document_id):
            delete_document_metadata(connection, document_id)
            raise RuntimeError("simulated metadata failure")

        with patch.object(database_mutations, "delete_document_metadata", side_effect=fail_metadata):
            with pytest.raises(RuntimeError, match="simulated metadata failure"):
                database_mutations.remove_document(tmp_path, 1)
    else:
        result = database_mutations.remove_document(tmp_path, 1)
        assert result["removed_chunks"] == 5
        assert result["cleanup_warning"] is None

    index = vector_search.FlatIndex(uri=str(tmp_path / "vector_index"))
    _distances, found = index.query(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), k=2)
    assert set(found[0].tolist()) == ({10, 20} if fail_after_delete else {60, 70})
