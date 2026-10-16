import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from core.constants import SUPPORTED_EXTENSIONS
from db.sqlite_operations import bump_version, record_addition, stored_database_info, upgrade_metadata_db

ALLOWED_EXTENSIONS = set(SUPPORTED_EXTENSIONS) | {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".json",
}


def _stage_files(paths, folder):
    staged = {}
    for raw_path in paths:
        source = Path(raw_path).resolve(strict=True)
        if not source.is_file() or source.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported or missing file: {raw_path}")
        if str(source) in staged.values():
            continue
        target = folder / source.name
        if target.exists() or target.is_symlink():
            import hashlib
            suffix = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:10]
            target = folder / f"{source.stem}_{suffix}{source.suffix}"
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)
        staged[str(target.resolve())] = str(source)
    return staged


def _restore_source_paths(prepared, staged):
    for metadata in [*(doc.metadata for doc in prepared.documents), *prepared.metadatas]:
        path = metadata.get("file_path")
        original = staged.get(str(Path(path).resolve())) if path else None
        if original:
            metadata["file_path"] = original
            metadata["file_name"] = Path(original).name


def _retain_new_documents(prepared, existing):
    kept = []
    skipped = []
    seen = set(existing)
    chunk_ids = {meta.get("doc_id") for meta in prepared.metadatas}
    for document in prepared.documents:
        meta = document.metadata
        key = (meta.get("hash"), meta.get("file_path"))
        if key in seen:
            skipped.append(meta.get("file_name", "unknown"))
        elif meta.get("doc_id") in chunk_ids:
            seen.add(key)
            kept.append(document)
    ids = {doc.metadata["doc_id"] for doc in kept}
    positions = [i for i, meta in enumerate(prepared.metadatas) if meta.get("doc_id") in ids]
    prepared.documents = kept
    prepared.texts = [prepared.texts[i] for i in positions]
    prepared.metadatas = [prepared.metadatas[i] for i in positions]
    prepared.vectors = prepared.vectors[positions]
    return skipped


def add_documents(database_name, database_directory, paths, progress=None):
    database_directory = Path(database_directory).resolve()
    metadata_path = database_directory / "metadata.db"
    if not metadata_path.is_file() or not (database_directory / "vectors").is_dir():
        raise FileNotFoundError(f"Database assets are missing in {database_directory}")
    if not paths:
        raise ValueError("Choose at least one file to add.")

    if progress:
        progress("Checking database")
    upgrade_metadata_db(database_directory)
    with closing(sqlite3.connect(metadata_path, timeout=30)) as connection:
        first_id = connection.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM document_metadata").fetchone()[0]
        existing = set(connection.execute("SELECT hash, file_path FROM document_metadata"))
        version = stored_database_info(connection)["version"]

    from db.database_interactions import append_chunks, prepare_documents

    with tempfile.TemporaryDirectory(prefix="vectordb_add_") as temporary:
        staged = _stage_files(paths, Path(temporary))
        if not staged:
            raise ValueError("No new files were selected.")
        if progress:
            progress("Extracting and embedding files")
        prepared = prepare_documents(database_name, temporary, first_document_id=first_id)
        _restore_source_paths(prepared, staged)
        skipped = _retain_new_documents(prepared, existing)
        if not prepared.documents:
            return {"added_documents": 0, "added_chunks": 0, "skipped": skipped, "not_added": prepared.not_added}

        if progress:
            progress("Writing new chunks")
        mappings = append_chunks(database_directory, prepared)
        ids = {doc.metadata["doc_id"] for doc in prepared.documents}
        if len(mappings) != len(prepared.texts) or {mapping[2] for mapping in mappings} != ids:
            raise RuntimeError("The appended chunks did not match the prepared documents.")

        if progress:
            progress("Saving document records")
        with closing(sqlite3.connect(metadata_path, timeout=30)) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if stored_database_info(connection)["version"] != version:
                    raise RuntimeError("The database changed while the new files were being prepared. Try again.")
                connection.executemany(
                    "INSERT INTO document_metadata (id, file_name, hash, file_path, page_content) VALUES (?, ?, ?, ?, ?)",
                    [(doc.metadata["doc_id"], doc.metadata.get("file_name"), doc.metadata.get("hash"),
                      doc.metadata.get("file_path"), doc.page_content) for doc in prepared.documents],
                )
                connection.executemany(
                    "INSERT INTO hash_chunk_ids (tiledb_id, hash, document_id) VALUES (?, ?, ?)", mappings,
                )
                record_addition(connection, ids)
                bump_version(connection)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return {
            "added_documents": len(prepared.documents),
            "added_chunks": len(mappings),
            "skipped": skipped,
            "not_added": prepared.not_added,
        }
