import sqlite3
import time
from pathlib import Path

import numpy as np

from db.sqlite_operations import (
    bump_version,
    chunk_ids_for_documents,
    delete_document_metadata,
    document_group,
    upgrade_metadata_db,
)

if not hasattr(np, "in1d"):
    np.in1d = np.isin


def _as_float_matrix(structured_vectors):
    values = np.ascontiguousarray(structured_vectors)
    return values.view(np.float32).reshape(len(values), -1)


def _clear_update_flag(index):
    index._set_has_updates(False)


def _clean_index(vector_search, index_uri, timestamp, index):
    errors = []
    try:
        vector_search.Index.clear_history(str(index_uri), timestamp)
    except Exception as error:
        errors.append(f"history cleanup failed: {error}")
    try:
        index.vacuum()
    except Exception as error:
        errors.append(f"vacuum failed: {error}")
    return errors


def _rollback_timestamp(previous):
    deadline = time.monotonic() + 1
    current = int(time.time() * 1000)
    while current <= previous:
        if time.monotonic() >= deadline:
            raise RuntimeError("The computer clock moved backward during the removal.")
        time.sleep(0.001)
        current = int(time.time() * 1000)
    return current


def remove_document(database_directory, document_id):
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

    index = vector_search.FlatIndex(uri=str(index_uri), open_for_remote_query_execution=True)
    mutation_timestamp = int(time.time() * 1000)
    if mutation_timestamp <= index.latest_ingestion_timestamp:
        raise RuntimeError(
            "The computer clock is earlier than this database's latest index update. "
            "Correct the clock before removing a document."
        )

    upgrade_metadata_db(database_directory)
    connection = sqlite3.connect(str(metadata_path), timeout=30)
    delete_attempted = False
    consolidated_index = None
    rollback_vectors = None
    chunk_ids = None

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

        delete_attempted = True
        index.delete_batch(external_ids=chunk_ids, timestamp=mutation_timestamp)

        delete_document_metadata(connection, document_id)
        bump_version(connection)
        remaining_vectors = connection.execute("SELECT COUNT(*) FROM hash_chunk_ids").fetchone()[0]
        consolidated_index = index.consolidate_updates()
        _clear_update_flag(consolidated_index)
        connection.commit()
        cleanup_errors = _clean_index(vector_search, index_uri, mutation_timestamp, index)

        return {
            "document_id": document_id,
            "removed_documents": document_count,
            "removed_chunks": len(chunk_ids),
            "remaining_vectors": remaining_vectors,
            "cleanup_warning": "; ".join(cleanup_errors) if cleanup_errors else None,
        }
    except Exception as original_error:
        connection.rollback()
        rollback_errors = []
        cleanup_errors = []
        if delete_attempted and rollback_vectors is not None:
            try:
                previous_timestamp = mutation_timestamp
                if consolidated_index is not None:
                    previous_timestamp = max(previous_timestamp, consolidated_index.latest_ingestion_timestamp)
                rollback_timestamp = _rollback_timestamp(previous_timestamp)
                index.update_batch(
                    vectors=rollback_vectors,
                    external_ids=chunk_ids,
                    timestamp=rollback_timestamp,
                )
                restored_index = index.consolidate_updates()
                _clear_update_flag(restored_index)
                cleanup_errors = _clean_index(vector_search, index_uri, rollback_timestamp, index)
            except Exception as rollback_error:
                rollback_errors.append(f"vector rollback failed: {rollback_error}")
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"Document removal failed and automatic rollback was incomplete ({details})."
            ) from original_error
        if cleanup_errors:
            details = "; ".join(cleanup_errors)
            raise RuntimeError(
                f"{original_error} The document was restored, but index cleanup could not finish ({details})."
            ) from original_error
        raise
    finally:
        connection.close()
        del index
