import sqlite3
from pathlib import Path


class DocumentNotFoundError(LookupError):
    """Raised when a requested document is not present in a metadata DB."""


def get_document_chunk_ids(connection, document_hash):
    """Return every TileDB external ID associated with one source document."""
    rows = connection.execute(
        "SELECT tiledb_id FROM hash_chunk_ids WHERE hash = ? ORDER BY tiledb_id",
        (document_hash,),
    ).fetchall()
    return [row[0] for row in rows]


def delete_document_metadata(connection, document_hash):
    """Delete a document and its chunk mapping inside the caller's transaction.

    The caller owns commit/rollback so this can be coordinated with the vector
    index mutation.
    """
    document_count = connection.execute(
        "SELECT COUNT(*) FROM document_metadata WHERE hash = ?",
        (document_hash,),
    ).fetchone()[0]
    if not document_count:
        raise DocumentNotFoundError(
            f"No document with hash '{document_hash}' exists in this database."
        )

    chunk_ids = get_document_chunk_ids(connection, document_hash)
    if not chunk_ids:
        raise RuntimeError(
            "The document has no vector mappings. The database may be inconsistent."
        )

    connection.execute("DELETE FROM hash_chunk_ids WHERE hash = ?", (document_hash,))
    connection.execute("DELETE FROM document_metadata WHERE hash = ?", (document_hash,))
    return chunk_ids


def create_metadata_db(persist_directory, documents, hash_id_mappings):
    if not persist_directory.exists():
        persist_directory.mkdir(parents=True, exist_ok=True)

    sqlite_db_path = persist_directory / "metadata.db"
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            hash TEXT,
            file_path TEXT,
            page_content TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hash_chunk_ids (
            tiledb_id TEXT PRIMARY KEY,
            hash TEXT
        )
    ''')

    try:
        doc_rows = [
            (
                doc.metadata.get("file_name", ""),
                doc.metadata.get("hash", ""),
                doc.metadata.get("file_path", ""),
                doc.page_content
            )
            for doc in documents
        ]
        cursor.executemany('''
            INSERT INTO document_metadata (file_name, hash, file_path, page_content)
            VALUES (?, ?, ?, ?)
        ''', doc_rows)

        cursor.executemany('''
            INSERT INTO hash_chunk_ids (tiledb_id, hash)
            VALUES (?, ?)
        ''', hash_id_mappings)

        conn.commit()
    finally:
        conn.close()
