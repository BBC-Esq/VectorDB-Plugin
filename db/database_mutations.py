import sqlite3
from pathlib import Path

from db.sqlite_operations import (
    bump_version,
    delete_document_metadata,
    document_group,
    record_removal,
    upgrade_metadata_db,
)


def remove_document(database_directory, document_id):
    database_directory = Path(database_directory).resolve()
    metadata_path = database_directory / "metadata.db"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Required database asset is missing: {metadata_path}")

    # Upgrading old databases can inspect TileDB to resolve duplicate files, so
    # this function runs in the mutation child process rather than the GUI.
    upgrade_metadata_db(database_directory)
    connection = sqlite3.connect(str(metadata_path), timeout=30)
    try:
        connection.execute("BEGIN IMMEDIATE")
        group = document_group(connection, document_id)
        # Capture chunk ownership while its metadata still exists.
        record_removal(connection, group)
        chunk_ids = delete_document_metadata(connection, document_id)
        bump_version(connection)
        remaining_vectors = connection.execute("SELECT COUNT(*) FROM hash_chunk_ids").fetchone()[0]
        connection.commit()
        return {
            "document_id": document_id,
            "removed_documents": len(group),
            "removed_chunks": len(chunk_ids),
            "remaining_vectors": remaining_vectors,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
