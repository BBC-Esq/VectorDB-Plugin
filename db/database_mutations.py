"""Transactional mutations for an existing vector database.

TileDB's vector index and the SQLite metadata DB are separate stores.  These
helpers coordinate both sides and compensate the vector update if a later
metadata operation fails, preventing deleted documents from becoming either
searchable ghosts or orphaned UI rows.
"""

import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

from db.sqlite_operations import delete_document_metadata, get_document_chunk_ids

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
    """Convert the fixed-width structured TileDB vector attribute to float32."""
    values = np.ascontiguousarray(structured_vectors)
    return values.view(np.float32).reshape(len(values), -1)


def remove_document(database_directory, document_hash):
    """Remove all chunks for one document from an existing database.

    Returns a small result dictionary suitable for a GUI or CLI response.
    """
    database_directory = Path(database_directory).resolve()
    metadata_path = database_directory / "metadata.db"
    vectors_uri = database_directory / "vectors"
    index_uri = database_directory / "vector_index"
    index_metadata_path = database_directory / "index_metadata.json"

    for required in (metadata_path, vectors_uri, index_uri, index_metadata_path):
        if not required.exists():
            raise FileNotFoundError(f"Required database asset is missing: {required}")

    import tiledb
    import tiledb.vector_search as vector_search

    connection = sqlite3.connect(str(metadata_path), timeout=30)
    old_index_metadata = None
    index = None
    delete_attempted = False
    mutation_timestamp = int(time.time() * 1000)

    try:
        connection.execute("BEGIN IMMEDIATE")
        chunk_id_strings = get_document_chunk_ids(connection, document_hash)
        if not chunk_id_strings:
            # Produces the more useful missing-document/inconsistent-data error.
            delete_document_metadata(connection, document_hash)
        document_count = connection.execute(
            "SELECT COUNT(*) FROM document_metadata WHERE hash = ?",
            (document_hash,),
        ).fetchone()[0]

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
        rollback_vectors = np.vstack([
            np.asarray(vector_by_id[int(vector_id)], dtype=np.float32)
            for vector_id in chunk_ids
        ])

        with open(index_metadata_path, "r", encoding="utf-8") as handle:
            old_index_metadata = json.load(handle)

        index = vector_search.FlatIndex(uri=str(index_uri))
        delete_attempted = True
        index.delete_batch(external_ids=chunk_ids, timestamp=mutation_timestamp)

        delete_document_metadata(connection, document_hash)
        new_index_metadata = dict(old_index_metadata)
        previous_count = int(old_index_metadata.get("num_vectors", 0))
        if previous_count < len(chunk_ids):
            raise RuntimeError(
                "The index metadata contains fewer vectors than this document maps to; "
                "no changes were made."
            )
        new_index_metadata["num_vectors"] = previous_count - len(chunk_ids)
        _atomic_write_json(index_metadata_path, new_index_metadata)
        connection.commit()

        return {
            "document_hash": document_hash,
            "removed_documents": document_count,
            "removed_chunks": len(chunk_ids),
            "remaining_vectors": new_index_metadata["num_vectors"],
        }
    except Exception as original_error:
        connection.rollback()
        rollback_errors = []
        if delete_attempted and index is not None:
            try:
                index.update_batch(
                    vectors=rollback_vectors,
                    external_ids=chunk_ids,
                    timestamp=mutation_timestamp + 1,
                )
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
