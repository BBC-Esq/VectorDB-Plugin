import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from db.sqlite_operations import (
    chunk_ids_for_documents,
    delete_document_metadata,
    document_group,
    ensure_document_ids,
)

if not hasattr(np, "in1d"):
    np.in1d = np.isin


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _as_float_matrix(structured_vectors):
    values = np.ascontiguousarray(structured_vectors)
    return values.view(np.float32).reshape(len(values), -1)


def remove_document(database_directory, document_id):
    database_directory = Path(database_directory).resolve()
    metadata_path = database_directory / "metadata.db"
    vectors_uri = database_directory / "vectors"
    index_uri = database_directory / "vector_index"
    index_metadata_path = database_directory / "index_metadata.json"

    for required in (metadata_path, vectors_uri, index_uri, index_metadata_path):
        if not required.exists():
            raise FileNotFoundError(f"Required database asset is missing: {required}")

    ensure_document_ids(database_directory)

    import tiledb
    import tiledb.vector_search as vector_search

    connection = sqlite3.connect(str(metadata_path), timeout=30)
    old_index_metadata = None
    index = None
    delete_attempted = False
    mutation_timestamp = int(time.time() * 1000)

    try:
        connection.execute("BEGIN IMMEDIATE")
        group = document_group(connection, document_id)
        if not group:
            delete_document_metadata(connection, document_id)
        digest = connection.execute(
            "SELECT hash FROM document_metadata WHERE id = ?", (document_id,)
        ).fetchone()[0]
        if connection.execute(
            "SELECT 1 FROM hash_chunk_ids WHERE hash = ? AND document_id IS NULL LIMIT 1", (digest,)
        ).fetchone():
            raise RuntimeError("Some chunks have no document owner. Removal was not started.")
        chunk_id_strings = chunk_ids_for_documents(connection, group)
        if not chunk_id_strings:
            raise RuntimeError("The document has no vector mappings. The database may be inconsistent.")
        document_count = len(group)

        chunk_ids = np.asarray(chunk_id_strings, dtype=np.uint64)
        with tiledb.open(str(vectors_uri), mode="r") as vector_array:
            raw = vector_array.multi_index[chunk_ids]
            returned_ids = np.asarray(raw["id"], dtype=np.uint64)
            returned_vectors = _as_float_matrix(raw["vector"])

        vector_by_id = {
            int(vector_id): returned_vectors[position]
            for position, vector_id in enumerate(returned_ids)
        }
        if any(int(vector_id) not in vector_by_id for vector_id in chunk_ids):
            raise RuntimeError(
                "One or more mapped vectors are missing from TileDB; no changes were made."
            )
        rollback_vectors = np.empty(len(chunk_ids), dtype=object)
        for position, vector_id in enumerate(chunk_ids):
            rollback_vectors[position] = np.asarray(vector_by_id[int(vector_id)], dtype=np.float32)

        with open(index_metadata_path, "r", encoding="utf-8") as handle:
            old_index_metadata = json.load(handle)

        index = vector_search.FlatIndex(uri=str(index_uri), open_for_remote_query_execution=True)
        delete_attempted = True
        index.delete_batch(external_ids=chunk_ids, timestamp=mutation_timestamp)

        delete_document_metadata(connection, document_id)
        new_index_metadata = dict(old_index_metadata)
        previous_count = int(old_index_metadata.get("num_vectors", 0))
        if previous_count < len(chunk_ids):
            raise RuntimeError(
                "The index metadata contains fewer vectors than this document maps to; "
                "no changes were made."
            )
        new_index_metadata["num_vectors"] = previous_count - len(chunk_ids)
        _atomic_write_json(index_metadata_path, new_index_metadata)
        consolidated_index = index.consolidate_updates()
        connection.commit()
        cleanup_errors = []
        try:
            vector_search.Index.clear_history(str(index_uri), mutation_timestamp)
        except Exception as error:
            cleanup_errors.append(f"history cleanup failed: {error}")
        try:
            consolidated_index.vacuum()
        except Exception as error:
            cleanup_errors.append(f"vacuum failed: {error}")

        return {
            "document_id": document_id,
            "removed_documents": document_count,
            "removed_chunks": len(chunk_ids),
            "remaining_vectors": new_index_metadata["num_vectors"],
            "cleanup_warning": "; ".join(cleanup_errors) if cleanup_errors else None,
        }
    except Exception as original_error:
        connection.rollback()
        rollback_errors = []
        if delete_attempted and index is not None:
            try:
                index.update_batch(
                    vectors=rollback_vectors,
                    external_ids=chunk_ids,
                    timestamp=max(int(time.time() * 1000), mutation_timestamp + 2),
                )
                index.consolidate_updates()
            except Exception as rollback_error:
                rollback_errors.append(f"vector rollback failed: {rollback_error}")
        if old_index_metadata is not None:
            try:
                _atomic_write_json(index_metadata_path, old_index_metadata)
            except Exception as rollback_error:
                rollback_errors.append(f"metadata rollback failed: {rollback_error}")
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"Document removal failed and automatic rollback was incomplete ({details})."
            ) from original_error
        raise
    finally:
        connection.close()
        del index
