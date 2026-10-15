import faulthandler
faulthandler.enable()

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from db.stage_write import _setup_tiledb_dlls, write_chunk_batches

RESULT_MARKER = "VECTORDB_APPEND_RESULT "


def unused_ids(tiledb, array_uri, rng):
    chosen = set()

    def choose(count):
        picked = []
        with tiledb.open(array_uri, mode="r") as array:
            existing = array.query(attrs=[], dims=["id"])
            while len(picked) < count:
                candidates = np.unique(rng.integers(low=0, high=np.iinfo(np.uint64).max, size=count - len(picked),
                                                    dtype=np.uint64, endpoint=False))
                taken = {int(value) for value in existing.multi_index[candidates]["id"]}
                for value in candidates.tolist():
                    if value not in taken and value not in chosen:
                        chosen.add(value)
                        picked.append(value)
        return np.array(picked, dtype=np.uint64)

    return choose


def append(database_directory, data_dir):
    database_directory = Path(database_directory)
    data_dir = Path(data_dir)
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    _setup_tiledb_dlls()
    import tiledb

    array_uri = str(database_directory / "vectors")
    if not Path(array_uri).is_dir():
        raise FileNotFoundError(f"The database has no chunk store: {array_uri}")
    with tiledb.open(array_uri, mode="r") as array:
        dimensions = array.schema.attr("vector").ncells
    if dimensions != int(meta["embedding_dim"]):
        raise ValueError(f"The new chunks have {meta['embedding_dim']} dimensions, but this database uses {dimensions}.")
    vectors_mm = np.load(str(data_dir / "vectors.npy"), mmap_mode="r")
    all_ids, hash_id_mappings = write_chunk_batches(array_uri, data_dir, meta, vectors_mm,
                                                    unused_ids(tiledb, array_uri, np.random.default_rng()))
    with open(data_dir / "hash_id_mappings.pkl", "wb") as f:
        pickle.dump(hash_id_mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(all_ids)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <database_directory> <data_directory>", file=sys.stderr)
        return 2
    chunks = append(sys.argv[1], sys.argv[2])
    print(RESULT_MARKER + json.dumps({"chunks": chunks}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
