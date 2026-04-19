import sqlite3
from pathlib import Path


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
