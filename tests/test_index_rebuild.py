import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest

from db import stage_rebuild
from db.sqlite_operations import (
    bump_version,
    create_metadata_db,
    database_stamp,
    pending_changes,
    record_addition,
    record_removal,
)
from db.text_splitter import Document

DIMENSIONS = 8
FILES = {1: ("a.txt", [101, 102, 103]), 2: ("b.txt", [201, 202]), 3: ("c.txt", [301])}


@pytest.fixture(scope="module")
def vector_search():
    stage_rebuild._setup_tiledb_dlls()
    import tiledb.vector_search as vs

    return vs


def _vector(chunk_id):
    return np.random.default_rng(chunk_id).random(DIMENSIONS, dtype=np.float32)


def build_database(folder, vector_search):
    import tiledb
    from tiledb.vector_search import _tiledbvspy as vspy

    folder.mkdir(parents=True)
    ids = np.array(sorted(chunk for _, chunks in FILES.values() for chunk in chunks), dtype=np.uint64)
    vectors = np.stack([_vector(int(chunk)) for chunk in ids])
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(tiledb.Dim(name="id", domain=(0, np.iinfo(np.uint64).max - 20000), tile=10000, dtype=np.uint64)),
        attrs=[
            tiledb.Attr(name="vector", dtype=np.dtype([("", np.float32)] * DIMENSIONS)),
            tiledb.Attr(name="text", dtype=str, var=True),
            tiledb.Attr(name="metadata", dtype=str, var=True),
        ],
        sparse=True,
    )
    tiledb.Array.create(str(folder / "vectors"), schema)
    owner = {chunk: name for name, chunks in FILES.values() for chunk in chunks}
    with tiledb.open(str(folder / "vectors"), mode="w") as array:
        array[ids] = {
            "vector": vectors.view([("", np.float32)] * DIMENSIONS).reshape(-1),
            "text": np.array([f"chunk {chunk}" for chunk in ids], dtype=object),
            "metadata": np.array([json.dumps({"file_name": owner[int(chunk)]}) for chunk in ids], dtype=object),
        }
    vector_search.ingest(index_type="FLAT", index_uri=str(folder / "vector_index"), input_vectors=vectors,
                         external_ids=ids, dimensions=DIMENSIONS, distance_metric=vspy.DistanceMetric.COSINE)
    (folder / "index_metadata.json").write_text(json.dumps({"distance_metric": "cosine", "dimensions": DIMENSIONS,
                                                            "vector_type": "float32", "index_type": "FLAT",
                                                            "num_vectors": len(ids)}), encoding="utf-8")
    documents = [Document(page_content=name, metadata={"doc_id": doc_id, "file_name": name, "hash": name, "file_path": f"C:/{name}"})
                 for doc_id, (name, _) in FILES.items()]
    mappings = [(str(chunk), name, doc_id) for doc_id, (name, chunks) in FILES.items() for chunk in chunks]
    create_metadata_db(folder, documents, mappings)
    return folder


def remove(folder, document_id):
    with closing(sqlite3.connect(folder / "metadata.db")) as connection:
        record_removal(connection, [document_id])
        connection.execute("DELETE FROM hash_chunk_ids WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM document_metadata WHERE id = ?", (document_id,))
        bump_version(connection)
        connection.commit()


def changes(folder):
    with closing(sqlite3.connect(folder / "metadata.db")) as connection:
        return pending_changes(connection)


def index_ids(folder):
    import tiledb

    with tiledb.Group(str(folder / "vector_index"), "r") as group:
        members = {member.name: member.uri for member in group}
        size = json.loads(group.meta["base_sizes"])[-1]
        stamp = json.loads(group.meta["ingestion_timestamps"])[-1]
    with tiledb.open(members["shuffled_vector_ids"], mode="r", timestamp=stamp) as array:
        data = array[0:size]
    return sorted(int(value) for value in np.asarray(data[next(iter(data))] if isinstance(data, dict) else data))


def store_ids(folder):
    import tiledb

    with tiledb.open(str(folder / "vectors"), mode="r") as array:
        return sorted(int(value) for value in array.query(attrs=["text"])[:]["id"])


def test_rebuild_keeps_only_the_current_chunks(tmp_path, vector_search):
    folder = build_database(tmp_path / "db", vector_search)
    created = database_stamp(folder)["created"]
    remove(folder, 1)
    assert changes(folder) == {"documents": 1, "removed_chunks": 3, "added_chunks": 0}
    result = stage_rebuild.rebuild(folder)
    assert (result["chunks"], result["changed_documents"], result["removed_chunks"]) == (3, 1, 3)
    assert index_ids(folder) == store_ids(folder) == [201, 202, 301]
    assert changes(folder) == {"documents": 0, "removed_chunks": 0, "added_chunks": 0}
    assert database_stamp(folder) == {"created": created, "version": 3}
    assert json.loads((folder / "index_metadata.json").read_text(encoding="utf-8"))["num_vectors"] == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == ["db"]
    assert not (folder / stage_rebuild.COMPLETE_FILE).exists()
    index = vector_search.FlatIndex(uri=str(folder / "vector_index"))
    _, found = index.query(np.array([_vector(202)]), k=1)
    assert int(found[0][0]) == 202


def test_failed_rebuild_changes_nothing(tmp_path, vector_search, monkeypatch):
    folder = build_database(tmp_path / "db", vector_search)
    remove(folder, 2)

    def broken(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(vector_search, "ingest", broken)
    with pytest.raises(RuntimeError, match="simulated failure"):
        stage_rebuild.rebuild(folder)
    assert changes(folder) == {"documents": 1, "removed_chunks": 2, "added_chunks": 0}
    assert database_stamp(folder)["version"] == 2
    assert index_ids(folder) == [101, 102, 103, 201, 202, 301]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["db"]


def test_rebuild_refuses_an_empty_database(tmp_path, vector_search):
    folder = build_database(tmp_path / "db", vector_search)
    for document_id in FILES:
        remove(folder, document_id)
    with pytest.raises(RuntimeError, match="no chunks left"):
        stage_rebuild.rebuild(folder)
    assert changes(folder) == {"documents": 3, "removed_chunks": 6, "added_chunks": 0}


def prepared_chunks(chunk_ids, document_id, name, dimensions=DIMENSIONS):
    from db.database_interactions import PreparedChunks

    metadata = {"file_name": name, "hash": name, "file_path": f"C:/{name}", "doc_id": document_id}
    vectors = np.stack([np.random.default_rng(chunk).random(dimensions, dtype=np.float32) for chunk in chunk_ids])
    return PreparedChunks([f"new chunk {chunk}" for chunk in chunk_ids], [dict(metadata) for _ in chunk_ids], vectors,
                          [Document(page_content=name, metadata=metadata)], [])


def test_added_chunks_are_appended_and_folded_in_by_a_rebuild(tmp_path, vector_search):
    from db.database_interactions import append_chunks

    folder = build_database(tmp_path / "db", vector_search)
    original = store_ids(folder)
    prepared = prepared_chunks([901, 902], 4, "d.txt")
    mappings = append_chunks(folder, prepared)
    new_ids = sorted(int(chunk_id) for chunk_id, _, _ in mappings)
    assert [(content_hash, document_id) for _, content_hash, document_id in mappings] == [("d.txt", 4), ("d.txt", 4)]
    assert len(new_ids) == 2 and not set(new_ids) & set(original)
    assert store_ids(folder) == sorted(original + new_ids)
    assert index_ids(folder) == original
    with closing(sqlite3.connect(folder / "metadata.db")) as connection:
        connection.execute("INSERT INTO document_metadata (id, file_name, hash, file_path, page_content) "
                           "VALUES (4, 'd.txt', 'd.txt', 'C:/d.txt', 'd.txt')")
        connection.executemany("INSERT INTO hash_chunk_ids (tiledb_id, hash, document_id) VALUES (?, ?, ?)", mappings)
        record_addition(connection, [4])
        bump_version(connection)
        connection.commit()
    assert changes(folder) == {"documents": 1, "removed_chunks": 0, "added_chunks": 2}
    result = stage_rebuild.rebuild(folder)
    assert (result["chunks"], result["changed_documents"], result["added_chunks"]) == (8, 1, 2)
    assert index_ids(folder) == store_ids(folder) == sorted(original + new_ids)
    assert changes(folder) == {"documents": 0, "removed_chunks": 0, "added_chunks": 0}
    index = vector_search.FlatIndex(uri=str(folder / "vector_index"))
    _, found = index.query(np.array([prepared.vectors[1]]), k=1)
    assert int(found[0][0]) == int(mappings[1][0])


def test_chunks_appended_without_a_commit_are_dropped_by_a_rebuild(tmp_path, vector_search):
    from db.database_interactions import append_chunks

    folder = build_database(tmp_path / "db", vector_search)
    original = store_ids(folder)
    append_chunks(folder, prepared_chunks([911], 4, "orphan.txt"))
    assert len(store_ids(folder)) == len(original) + 1
    stage_rebuild.rebuild(folder)
    assert index_ids(folder) == store_ids(folder) == original


def test_appending_vectors_of_another_size_is_refused(tmp_path, vector_search):
    from db.database_interactions import append_chunks

    folder = build_database(tmp_path / "db", vector_search)
    original = store_ids(folder)
    with pytest.raises(RuntimeError, match="dimensions"):
        append_chunks(folder, prepared_chunks([921], 4, "wide.txt", dimensions=DIMENSIONS + 4))
    assert store_ids(folder) == original


def make_folder(path, marker=None):
    path.mkdir(parents=True)
    (path / "metadata.db").write_text(path.name, encoding="utf-8")
    if marker:
        (path / stage_rebuild.COMPLETE_FILE).write_text("complete", encoding="utf-8")
    return path


def test_an_interrupted_swap_is_finished(tmp_path):
    live = tmp_path / "db"
    make_folder(stage_rebuild.staging_folder(live), marker=True)
    make_folder(stage_rebuild.previous_folder(live))
    stage_rebuild.finish_interrupted_rebuild(live)
    assert (live / "metadata.db").read_text(encoding="utf-8") == ".rebuild_db"
    assert not (live / stage_rebuild.COMPLETE_FILE).exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["db"]


def test_an_unfinished_rebuild_is_undone(tmp_path):
    live = tmp_path / "db"
    make_folder(stage_rebuild.staging_folder(live))
    make_folder(stage_rebuild.previous_folder(live))
    stage_rebuild.finish_interrupted_rebuild(live)
    assert (live / "metadata.db").read_text(encoding="utf-8") == ".rebuilt_db"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["db"]


def test_leftovers_are_cleaned_up_at_startup(tmp_path):
    make_folder(tmp_path / "kept")
    make_folder(stage_rebuild.staging_folder(tmp_path / "kept"))
    make_folder(tmp_path / "done")
    make_folder(stage_rebuild.previous_folder(tmp_path / "done"))
    make_folder(stage_rebuild.staging_folder(tmp_path / "swapped"), marker=True)
    make_folder(stage_rebuild.previous_folder(tmp_path / "swapped"))
    stage_rebuild.finish_interrupted_rebuilds(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["done", "kept", "swapped"]
    assert (tmp_path / "kept" / "metadata.db").read_text(encoding="utf-8") == "kept"
    assert (tmp_path / "swapped" / "metadata.db").read_text(encoding="utf-8") == ".rebuild_swapped"
