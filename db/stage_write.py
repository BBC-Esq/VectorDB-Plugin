"""
Stage 5 (isolated): write the TileDB array + FLAT index in a FRESH subprocess.

The GPU forward pass fragments the build process's heap, which is what makes the
large-scale TileDB write loop crash (heap corruption / access violation). This
script runs that same write on a clean, unfragmented heap: it reads the handoff
data from disk (embeddings as a .npy memmap, texts + metadata as per-batch
pickle shards) and streams it batch-by-batch, so peak memory stays ~one batch.

Self-contained on purpose: imports only numpy + tiledb (+ optional orjson) and
does NOT import torch or db.database_interactions, to keep this process pristine.

Usage:
    python stage_write.py <persist_dir> <data_dir>

<data_dir> must contain:
    meta.json        {"embedding_dim", "num_vectors", "batch_size", "num_batches"}
    vectors.npy      float32 (num_vectors, embedding_dim)
    shard_NNNNN.pkl  {"texts": [...], "metadatas": [...]} for each batch

On success it writes:
    <persist_dir>/vectors/           TileDB sparse array
    <persist_dir>/vector_index/      FLAT index
    <persist_dir>/index_metadata.json
    <data_dir>/hash_id_mappings.pkl  list[(tiledb_id_str, file_hash)]
"""
import faulthandler
faulthandler.enable()

import ctypes
import gc
import json
import logging
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("stage_write")

try:
    import orjson

    def _json_dumps(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")
except ImportError:
    def _json_dumps(obj) -> str:
        return json.dumps(obj)


def _setup_tiledb_dlls():
    import tiledb  # noqa: F401

    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    site_packages = os.path.join(venv_root, "Lib", "site-packages")
    tiledb_libs = os.path.join(site_packages, "tiledb.libs")
    vector_search_lib = os.path.join(site_packages, "tiledb", "vector_search", "lib")

    for directory in [tiledb_libs, vector_search_lib]:
        if os.path.isdir(directory):
            try:
                os.add_dll_directory(directory)
            except OSError:
                pass

    if os.path.isdir(tiledb_libs):
        for filename in sorted(os.listdir(tiledb_libs)):
            if filename.endswith(".dll"):
                try:
                    ctypes.CDLL(os.path.join(tiledb_libs, filename))
                except Exception:
                    pass

    if os.path.isdir(vector_search_lib):
        tiledb_dll = os.path.join(vector_search_lib, "tiledb.dll")
        if os.path.exists(tiledb_dll):
            try:
                ctypes.CDLL(tiledb_dll)
            except Exception:
                pass


def main():
    persist_dir = Path(sys.argv[1])
    data_dir = Path(sys.argv[2])

    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    embedding_dim = int(meta["embedding_dim"])
    num_vectors = int(meta["num_vectors"])
    batch_size = int(meta["batch_size"])
    num_batches = int(meta["num_batches"])

    _setup_tiledb_dlls()
    import tiledb
    import tiledb.vector_search as vs
    from tiledb.vector_search import _tiledbvspy as vspy

    array_uri = str(persist_dir / "vectors")
    index_uri = str(persist_dir / "vector_index")

    for partial in (array_uri, index_uri):
        if os.path.isdir(partial):
            shutil.rmtree(partial, ignore_errors=True)

    t0 = time.time()
    logger.info(f"Creating TileDB array: {num_vectors:,} vectors of dimension {embedding_dim}")

    dom = tiledb.Domain(
        tiledb.Dim(name="id", domain=(0, np.iinfo(np.uint64).max - 20000), tile=10000, dtype=np.uint64)
    )
    attrs = [
        tiledb.Attr(name="vector", dtype=np.dtype([("", np.float32)] * embedding_dim)),
        tiledb.Attr(name="text", dtype=str, var=True),
        tiledb.Attr(name="metadata", dtype=str, var=True),
    ]
    schema = tiledb.ArraySchema(
        domain=dom, attrs=attrs, sparse=True,
        cell_order="row-major", tile_order="row-major",
    )
    tiledb.Array.create(array_uri, schema)

    vectors_mm = np.load(str(data_dir / "vectors.npy"), mmap_mode="r")
    logger.info(f"Writing TileDB array in {num_batches} batch(es)")

    all_ids = np.empty(num_vectors, dtype=np.uint64)
    hash_id_mappings = []
    rng = np.random.default_rng()

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_vectors)

        with open(data_dir / f"shard_{batch_idx:05d}.pkl", "rb") as f:
            shard = pickle.load(f)
        batch_texts_list = shard["texts"]
        batch_metas = shard["metadatas"]
        del shard

        batch_ids = rng.integers(
            low=0, high=np.iinfo(np.uint64).max,
            size=end - start, dtype=np.uint64, endpoint=False,
        )
        all_ids[start:end] = batch_ids

        batch_id_strs = batch_ids.astype(str).tolist()
        for j in range(end - start):
            md = batch_metas[j]
            file_hash = md.get("hash", "") if isinstance(md, dict) else ""
            hash_id_mappings.append((batch_id_strs[j], file_hash))

        batch_meta_strs = []
        prev_meta = None
        prev_meta_str = None
        for md in batch_metas:
            if prev_meta_str is not None and md == prev_meta:
                batch_meta_strs.append(prev_meta_str)
            else:
                prev_meta_str = _json_dumps(md)
                prev_meta = md
                batch_meta_strs.append(prev_meta_str)

        batch_texts = np.array(batch_texts_list, dtype=object)
        batch_metadata = np.array(batch_meta_strs, dtype=object)

        batch_vectors = np.ascontiguousarray(vectors_mm[start:end])
        batch_structured = batch_vectors.view([("", np.float32)] * embedding_dim).reshape(-1)

        with tiledb.open(array_uri, mode="w") as A:
            A[batch_ids] = {
                "vector": batch_structured,
                "text": batch_texts,
                "metadata": batch_metadata,
            }

        del batch_structured, batch_vectors, batch_texts, batch_metadata
        del batch_texts_list, batch_metas, batch_meta_strs, batch_id_strs
        gc.collect()
        logger.info(f"  batch {batch_idx + 1}/{num_batches} written ({end:,}/{num_vectors:,})")

    tiledb.consolidate(array_uri)
    tiledb.vacuum(array_uri)

    logger.info("Building FLAT index...")
    vs.ingest(
        index_type="FLAT",
        index_uri=index_uri,
        input_vectors=np.array(vectors_mm),
        external_ids=all_ids,
        dimensions=embedding_dim,
        distance_metric=vspy.DistanceMetric.COSINE,
    )

    with open(persist_dir / "index_metadata.json", "w") as f:
        json.dump({
            "distance_metric": "cosine",
            "dimensions": embedding_dim,
            "vector_type": "float32",
            "index_type": "FLAT",
            "num_vectors": num_vectors,
        }, f)

    with open(data_dir / "hash_id_mappings.pkl", "wb") as f:
        pickle.dump(hash_id_mappings, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"FLAT index created at: {index_uri}")
    logger.info(f"Write stage complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
