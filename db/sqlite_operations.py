import sqlite3
from pathlib import Path

METADATA_SCHEMA_VERSION = 2


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
            hash TEXT,
            document_id INTEGER
        )
    ''')

    try:
        doc_ids = [doc.metadata.get("doc_id") for doc in documents]
        use_doc_ids = all(isinstance(doc_id, int) for doc_id in doc_ids) and len(set(doc_ids)) == len(doc_ids)
        doc_rows = [
            (
                doc_id if use_doc_ids else None,
                doc.metadata.get("file_name", ""),
                doc.metadata.get("hash", ""),
                doc.metadata.get("file_path", ""),
                doc.page_content
            )
            for doc, doc_id in zip(documents, doc_ids)
        ]
        cursor.executemany('''
            INSERT INTO document_metadata (id, file_name, hash, file_path, page_content)
            VALUES (?, ?, ?, ?, ?)
        ''', doc_rows)

        cursor.executemany('''
            INSERT INTO hash_chunk_ids (tiledb_id, hash, document_id)
            VALUES (?, ?, ?)
        ''', (
            (mapping[0], mapping[1], mapping[2] if use_doc_ids and len(mapping) > 2 else None)
            for mapping in hash_id_mappings
        ))

        _assign_document_ids(conn, persist_directory / "vectors")
        conn.execute(f"PRAGMA user_version = {METADATA_SCHEMA_VERSION}")
        conn.commit()
    finally:
        conn.close()


def ensure_document_ids(database_directory):
    database_directory = Path(database_directory)
    connection = sqlite3.connect(database_directory / "metadata.db", timeout=30, isolation_level=None)
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] >= METADATA_SCHEMA_VERSION:
            return False
        connection.execute("BEGIN IMMEDIATE")
        try:
            if connection.execute("PRAGMA user_version").fetchone()[0] >= METADATA_SCHEMA_VERSION:
                connection.execute("ROLLBACK")
                return False
            columns = {row[1] for row in connection.execute("PRAGMA table_info(hash_chunk_ids)")}
            if "document_id" not in columns:
                connection.execute("ALTER TABLE hash_chunk_ids ADD COLUMN document_id INTEGER")
            _assign_document_ids(connection, database_directory / "vectors")
            connection.execute(f"PRAGMA user_version = {METADATA_SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        return True
    finally:
        connection.close()


def document_group(connection, document_id):
    row = connection.execute(
        "SELECT hash, file_path FROM document_metadata WHERE id = ?", (document_id,)
    ).fetchone()
    if row is None:
        return []
    return [doc_id for (doc_id,) in connection.execute(
        "SELECT id FROM document_metadata WHERE hash IS ? AND file_path IS ? ORDER BY id", row
    )]


def documents_with_content(connection, content_hash):
    return [doc_id for (doc_id,) in connection.execute(
        "SELECT id FROM document_metadata WHERE hash = ? ORDER BY id", (content_hash,)
    )]


def chunk_ids_for_documents(connection, document_ids):
    document_ids = list(document_ids)
    chunk_ids = []
    for start in range(0, len(document_ids), 500):
        batch = document_ids[start:start + 500]
        placeholders = ", ".join("?" * len(batch))
        chunk_ids.extend(tiledb_id for (tiledb_id,) in connection.execute(
            f"SELECT tiledb_id FROM hash_chunk_ids WHERE document_id IN ({placeholders})", batch
        ))
    return chunk_ids


def _assign_document_ids(connection, vectors_uri):
    connection.execute("CREATE INDEX IF NOT EXISTS document_metadata_hash ON document_metadata(hash)")
    connection.execute("DROP TABLE IF EXISTS temp.single_documents")
    connection.execute(
        "CREATE TEMP TABLE single_documents AS "
        "SELECT hash, MIN(id) AS id FROM document_metadata GROUP BY hash HAVING COUNT(*) = 1"
    )
    connection.execute("CREATE INDEX temp.single_documents_hash ON single_documents(hash)")
    connection.execute(
        "UPDATE hash_chunk_ids SET document_id = single_documents.id FROM single_documents "
        "WHERE hash_chunk_ids.document_id IS NULL AND hash_chunk_ids.hash = single_documents.hash"
    )
    connection.execute("DROP TABLE temp.single_documents")

    shared_hashes = "SELECT hash FROM document_metadata GROUP BY hash HAVING COUNT(*) > 1"
    shared_chunks = connection.execute(
        f"SELECT tiledb_id, hash FROM hash_chunk_ids WHERE document_id IS NULL AND hash IN ({shared_hashes})"
    ).fetchall()
    if shared_chunks and Path(vectors_uri).is_dir():
        chunk_paths = _chunk_file_paths(vectors_uri, [tiledb_id for tiledb_id, _ in shared_chunks])
        owners = {}
        for doc_id, content_hash, file_path in connection.execute(
            f"SELECT id, hash, file_path FROM document_metadata WHERE hash IN ({shared_hashes})"
        ):
            key = (content_hash, file_path)
            owners[key] = min(doc_id, owners.get(key, doc_id))
        connection.executemany(
            "UPDATE hash_chunk_ids SET document_id = ? WHERE tiledb_id = ?",
            [
                (owners[(content_hash, chunk_paths.get(tiledb_id))], tiledb_id)
                for tiledb_id, content_hash in shared_chunks
                if (content_hash, chunk_paths.get(tiledb_id)) in owners
            ],
        )

    connection.execute("CREATE INDEX IF NOT EXISTS hash_chunk_ids_document ON hash_chunk_ids(document_id)")


def _chunk_file_paths(vectors_uri, tiledb_ids):
    import json

    import numpy as np
    import tiledb

    chunk_paths = {}
    with tiledb.open(str(vectors_uri), mode="r") as array:
        query = array.query(attrs=["metadata"])
        for start in range(0, len(tiledb_ids), 10000):
            batch = np.array([int(value) for value in tiledb_ids[start:start + 10000]], dtype=np.uint64)
            data = query.multi_index[batch]
            for chunk_id, raw in zip(data["id"], data["metadata"]):
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                try:
                    chunk_paths[str(int(chunk_id))] = json.loads(text).get("file_path")
                except (ValueError, AttributeError):
                    continue
    return chunk_paths
