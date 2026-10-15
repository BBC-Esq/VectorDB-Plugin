import ctypes
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.sqlite_operations import bump_version, clear_changes, pending_changes, upgrade_metadata_db

RESULT_MARKER = "VECTORDB_REBUILD_RESULT "
STAGING_PREFIX = ".rebuild_"
PREVIOUS_PREFIX = ".rebuilt_"
COMPLETE_FILE = "REBUILD_COMPLETE"
MANAGED = {"metadata.db", "vectors", "vector_index", "index_metadata.json", COMPLETE_FILE}
BATCH_SIZE = 20000


def _setup_tiledb_dlls():
    __import__("tiledb")
    site_packages = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Lib", "site-packages")
    folders = (os.path.join(site_packages, "tiledb.libs"), os.path.join(site_packages, "tiledb", "vector_search", "lib"))
    for folder in folders:
        if os.path.isdir(folder):
            try:
                os.add_dll_directory(folder)
            except OSError:
                pass
    for folder in folders:
        if os.path.isdir(folder):
            for filename in sorted(os.listdir(folder)):
                if filename.endswith(".dll"):
                    try:
                        ctypes.CDLL(os.path.join(folder, filename))
                    except Exception:
                        pass


def staging_folder(database_directory):
    database_directory = Path(database_directory)
    return database_directory.parent / f"{STAGING_PREFIX}{database_directory.name}"


def previous_folder(database_directory):
    database_directory = Path(database_directory)
    return database_directory.parent / f"{PREVIOUS_PREFIX}{database_directory.name}"


def _rename(source, target, attempts=10):
    for attempt in range(attempts):
        try:
            os.rename(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5)


def finish_interrupted_rebuild(database_directory):
    database_directory = Path(database_directory)
    staging = staging_folder(database_directory)
    previous = previous_folder(database_directory)
    if not database_directory.exists():
        if (staging / COMPLETE_FILE).exists():
            _rename(staging, database_directory)
        elif previous.exists():
            _rename(previous, database_directory)
    (database_directory / COMPLETE_FILE).unlink(missing_ok=True)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    if previous.exists() and database_directory.exists():
        shutil.rmtree(previous, ignore_errors=True)


def finish_interrupted_rebuilds(vector_root):
    try:
        with os.scandir(vector_root) as entries:
            names = [entry.name for entry in entries if entry.is_dir()]
    except OSError:
        return
    for name in names:
        for prefix in (STAGING_PREFIX, PREVIOUS_PREFIX):
            if name.startswith(prefix) and name[len(prefix):]:
                finish_interrupted_rebuild(Path(vector_root) / name[len(prefix):])


def _copy_chunks(old_uri, new_uri, chunk_ids, dimensions):
    import numpy as np
    import tiledb

    tiledb.Array.create(new_uri, tiledb.ArraySchema.load(old_uri))
    vectors = np.empty((len(chunk_ids), dimensions), dtype=np.float32)
    ids = np.empty(len(chunk_ids), dtype=np.uint64)
    position = 0
    with tiledb.open(old_uri, mode="r") as source:
        for start in range(0, len(chunk_ids), BATCH_SIZE):
            batch = chunk_ids[start:start + BATCH_SIZE]
            data = source.multi_index[batch]
            found = np.asarray(data["id"], dtype=np.uint64)
            if len(found) != len(batch):
                raise RuntimeError(f"{len(batch) - len(found):,} of the database's chunks are missing from its chunk store, "
                                   "so the index was not rebuilt.")
            with tiledb.open(new_uri, mode="w") as target:
                target[found] = {"vector": data["vector"], "text": data["text"], "metadata": data["metadata"]}
            vectors[position:position + len(found)] = np.ascontiguousarray(data["vector"]).view(np.float32).reshape(len(found), dimensions)
            ids[position:position + len(found)] = found
            position += len(found)
    tiledb.consolidate(new_uri)
    tiledb.vacuum(new_uri)
    return vectors, ids


def _copy_metadata(source_path, target_path):
    source = sqlite3.connect(source_path)
    target = sqlite3.connect(target_path)
    try:
        source.backup(target)
        clear_changes(target)
        bump_version(target)
        target.commit()
        target.execute("VACUUM")
    finally:
        target.close()
        source.close()


def rebuild(database_directory):
    started = time.time()
    database_directory = Path(database_directory).resolve()
    finish_interrupted_rebuild(database_directory)
    for required in ("metadata.db", "vectors", "vector_index", "index_metadata.json"):
        if not (database_directory / required).exists():
            raise FileNotFoundError(f"Required database file is missing: {database_directory / required}")
    upgrade_metadata_db(database_directory)

    connection = sqlite3.connect((database_directory / "metadata.db").resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        chunk_ids = sorted(int(tiledb_id) for (tiledb_id,) in connection.execute("SELECT tiledb_id FROM hash_chunk_ids"))
        pending = pending_changes(connection)
    finally:
        connection.close()
    if not chunk_ids:
        raise RuntimeError("The database has no chunks left, so there is no index to rebuild. Delete the database instead.")
    index_metadata = json.loads((database_directory / "index_metadata.json").read_text(encoding="utf-8"))
    dimensions = int(index_metadata["dimensions"])

    _setup_tiledb_dlls()
    import numpy as np
    import tiledb.vector_search as vs
    from tiledb.vector_search import _tiledbvspy as vspy

    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    staging = staging_folder(database_directory)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        vectors, ids = _copy_chunks(str(database_directory / "vectors"), str(staging / "vectors"),
                                    np.asarray(chunk_ids, dtype=np.uint64), dimensions)
        vs.ingest(
            index_type="FLAT",
            index_uri=str(staging / "vector_index"),
            input_vectors=vectors,
            external_ids=ids,
            dimensions=dimensions,
            distance_metric=vspy.DistanceMetric.COSINE,
        )
        del vectors
        index_metadata["num_vectors"] = len(chunk_ids)
        (staging / "index_metadata.json").write_text(json.dumps(index_metadata), encoding="utf-8")
        _copy_metadata(database_directory / "metadata.db", staging / "metadata.db")
        for entry in database_directory.iterdir():
            if entry.name not in MANAGED:
                if entry.is_dir():
                    shutil.copytree(entry, staging / entry.name)
                else:
                    shutil.copy2(entry, staging / entry.name)
        (staging / COMPLETE_FILE).write_text("complete", encoding="utf-8")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    previous = previous_folder(database_directory)
    try:
        if previous.exists():
            shutil.rmtree(previous)
        _rename(database_directory, previous)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        _rename(staging, database_directory)
    except BaseException:
        _rename(previous, database_directory)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    (database_directory / COMPLETE_FILE).unlink(missing_ok=True)
    shutil.rmtree(previous, ignore_errors=True)
    return {
        "chunks": len(chunk_ids),
        "changed_documents": pending["documents"],
        "removed_chunks": pending["removed_chunks"],
        "added_chunks": pending["added_chunks"],
        "seconds": round(time.time() - started, 1),
    }


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <database_directory>", file=sys.stderr)
        return 2
    result = rebuild(sys.argv[1])
    print(RESULT_MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
