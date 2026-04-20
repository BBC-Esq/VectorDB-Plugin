import faulthandler
faulthandler.enable()

# Module-level TileDB DLL preload. Mirrors the approach in VectorDB-Light's
# vector_db_query.py (lines 1-31). Critical for subprocesses spawned via
# multiprocessing.Process (e.g. the chunks-only query path): when the
# fresh interpreter imports this module, the DLLs load immediately —
# before any other code can accidentally trigger a tiledb.vector_search
# import without DLL registration, which on Windows causes the
# _tiledbvspy native module to fail with
#   ImportError: DLL load failed while importing _tiledbvspy
# or an even worse silent hang. The standalone _setup_tiledb_dlls()
# function below is kept for the creation subprocess path, where DLL
# setup has to happen after configure_logging() / set_cuda_paths().
import os
import sys
import ctypes

try:
    import tiledb as _tiledb_bootstrap  # noqa: F401

    _venv_root = os.path.dirname(os.path.dirname(sys.executable))
    _site_packages = os.path.join(_venv_root, "Lib", "site-packages")
    _tiledb_libs = os.path.join(_site_packages, "tiledb.libs")
    _vector_search_lib = os.path.join(_site_packages, "tiledb", "vector_search", "lib")

    for _directory in (_tiledb_libs, _vector_search_lib):
        if os.path.isdir(_directory):
            try:
                os.add_dll_directory(_directory)
            except OSError:
                pass

    if os.path.isdir(_tiledb_libs):
        for _filename in sorted(os.listdir(_tiledb_libs)):
            if _filename.endswith(".dll"):
                try:
                    ctypes.CDLL(os.path.join(_tiledb_libs, _filename))
                except Exception:
                    pass

    if os.path.isdir(_vector_search_lib):
        _tiledb_dll = os.path.join(_vector_search_lib, "tiledb.dll")
        if os.path.exists(_tiledb_dll):
            try:
                ctypes.CDLL(_tiledb_dll)
            except Exception:
                pass
except ImportError:
    # tiledb not installed — will fail later at actual use. Don't block
    # the import itself in case this module is loaded for non-TileDB work
    # (e.g. tests that only exercise pure helpers).
    pass

import gc
import json
import logging
import pickle
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# orjson is a Rust-based JSON encoder that's ~10x faster than stdlib json
# and avoids the heap fragmentation that triggers OverflowError + access
# violation when serializing millions of metadata dicts in tight loops.
try:
    import orjson

    def _json_dumps(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")
except ImportError:
    def _json_dumps(obj) -> str:
        return json.dumps(obj)

from db.document_processor import Document
from db.embedding_models import load_embedding_model
from db.sqlite_operations import create_metadata_db
from db.cuda_manager import get_cuda_manager
from core.config import get_config
from core.constants import PROJECT_ROOT, PIPELINE_PRESETS
from core.utilities import my_cprint, set_cuda_paths, configure_logging

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RUST_BACKTRACE", "1")

STAGE_EXTRACT_PATH = PROJECT_ROOT / "db" / "stage_extract.py"
STAGE_SPLIT_PATH = PROJECT_ROOT / "db" / "stage_split.py"

EXTRACT_MAX_RETRIES = 3
SPLIT_MAX_WORKER_RETRIES = 3
SPLIT_MAX_RETRIES = 5
TILEDB_WRITE_BATCH_SIZE = 100000

MAX_UINT64_SENTINEL = np.iinfo(np.uint64).max


def _get_split_params():
    try:
        preset_name = get_config().database.pipeline_preset
    except Exception:
        preset_name = "normal"
    preset = PIPELINE_PRESETS.get(preset_name, PIPELINE_PRESETS["normal"])
    return preset["split_max_parallel_workers"], preset["split_worker_batch_size"]


def _run_subprocess_stage(name, cmd, timeout=3600):
    logger.info(f"Starting subprocess stage: {name}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    output_lines = []
    for line in process.stdout:
        line = line.rstrip("\n")
        if line.strip():
            logger.info(f"  [{name}] {line}")
            output_lines.append(line)

    process.wait(timeout=timeout)

    if process.returncode != 0:
        for line in output_lines[-10:]:
            logger.error(f"  {line}")

    return process.returncode, output_lines


def _run_extract_with_retry(source_dir, output_pkl):
    python = sys.executable
    cmd = [python, str(STAGE_EXTRACT_PATH), str(source_dir), str(output_pkl)]

    for attempt in range(1, EXTRACT_MAX_RETRIES + 1):
        logger.info(f"Extract attempt {attempt}/{EXTRACT_MAX_RETRIES}")
        exit_code, _ = _run_subprocess_stage(f"Extract (attempt {attempt})", cmd)

        if exit_code == 0 and output_pkl.exists():
            logger.info(f"Extract stage completed on attempt {attempt}")
            return

        logger.error(f"Extract attempt {attempt} failed (exit code {exit_code})")

        if attempt < EXTRACT_MAX_RETRIES:
            logger.info("Waiting 3 seconds before retry...")
            time.sleep(3)
            gc.collect()

    raise RuntimeError(f"Extract stage failed after {EXTRACT_MAX_RETRIES} attempts")


def _run_split_with_retry(extracted_pkl, chunks_pkl, chunk_size, chunk_overlap, checkpoint_dir):
    python = sys.executable
    split_parallel, split_batch = _get_split_params()

    for attempt in range(1, SPLIT_MAX_RETRIES + 1):
        logger.info(f"Split attempt {attempt}/{SPLIT_MAX_RETRIES}")

        split_cmd = [
            python, str(STAGE_SPLIT_PATH),
            str(extracted_pkl),
            str(chunks_pkl),
            str(chunk_size),
            str(chunk_overlap),
            "--worker-batch-size", str(split_batch),
            "--max-worker-retries", str(SPLIT_MAX_WORKER_RETRIES),
            "--max-parallel-workers", str(split_parallel),
            "--checkpoint-dir", str(checkpoint_dir),
            "--checkpoint-interval", "5",
        ]

        exit_code, _ = _run_subprocess_stage(f"Split (attempt {attempt})", split_cmd)

        if exit_code == 0 and chunks_pkl.exists():
            logger.info(f"Split stage completed on attempt {attempt}")
            return

        logger.error(f"Split attempt {attempt} failed (exit code {exit_code})")

        if attempt < SPLIT_MAX_RETRIES:
            logger.info("Waiting 3 seconds before retry...")
            time.sleep(3)
            gc.collect()

    raise RuntimeError(f"Split stage failed after {SPLIT_MAX_RETRIES} attempts")


def _setup_tiledb_dlls():
    import ctypes
    import tiledb

    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    site_packages = os.path.join(venv_root, 'Lib', 'site-packages')

    tiledb_libs = os.path.join(site_packages, 'tiledb.libs')
    vector_search_lib = os.path.join(site_packages, 'tiledb', 'vector_search', 'lib')

    for directory in [tiledb_libs, vector_search_lib]:
        if os.path.isdir(directory):
            try:
                os.add_dll_directory(directory)
            except OSError:
                pass

    if os.path.isdir(tiledb_libs):
        for filename in sorted(os.listdir(tiledb_libs)):
            if filename.endswith('.dll'):
                try:
                    ctypes.CDLL(os.path.join(tiledb_libs, filename))
                except Exception:
                    pass

    if os.path.isdir(vector_search_lib):
        tiledb_dll = os.path.join(vector_search_lib, 'tiledb.dll')
        if os.path.exists(tiledb_dll):
            try:
                ctypes.CDLL(tiledb_dll)
            except Exception:
                pass


def create_vector_db_in_process(database_name):
    faulthandler.enable()
    configure_logging("INFO")
    set_cuda_paths()
    _setup_tiledb_dlls()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RUST_BACKTRACE"] = "1"

    create_vector_db = None

    try:
        create_vector_db = CreateVectorDB(database_name=database_name)
        create_vector_db.run()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if create_vector_db:
            del create_vector_db

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

        time.sleep(0.1)


def process_chunks_only_query(database_name, query, result_queue):
    configure_logging("INFO")
    try:
        query_db = QueryVectorDB(database_name)
        try:
            contexts, metadata_list = query_db.search(query)

            if not contexts:
                result_queue.put(
                    "No chunks passed the similarity threshold.\n\n"
                    "Try lowering the 'Similarity' setting in the Database Query "
                    "settings tab (e.g. from 0.7 to 0.4) and run the query again."
                )
                return

            formatted_contexts = []
            for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
                file_name = metadata.get('file_name', 'Unknown')
                cleaned_context = re.sub(r'\n[ \t]+\n', '\n\n', context)
                cleaned_context = re.sub(r'\n\s*\n\s*\n*', '\n\n', cleaned_context.strip())
                formatted_context = (
                    f"{'-'*80}\n"
                    f"CONTEXT {index} | {file_name}\n"
                    f"{'-'*80}\n"
                    f"{cleaned_context}\n"
                )
                formatted_contexts.append(formatted_context)

            result_queue.put("\n".join(formatted_contexts))
        finally:
            query_db.close()
    except Exception as e:
        result_queue.put(f"Error querying database: {str(e)}")


class CreateVectorDB:
    def __init__(self, database_name):
        self.config = get_config()
        self.SOURCE_DIRECTORY = self.config.docs_dir
        self.PERSIST_DIRECTORY = self.config.vector_db_dir / database_name

    @torch.inference_mode()
    def initialize_vector_model(self, embedding_model_name, config_data):
        return load_embedding_model(
            model_path=embedding_model_name,
            compute_device=config_data.Compute_Device.database_creation,
            use_half=config_data.database.half,
            is_query=False,
            verbose=True,
        )

    def _create_tiledb_array(self, texts, vectors_array, metadatas):
        _setup_tiledb_dlls()

        import tiledb
        import tiledb.vector_search as vs
        from tiledb.vector_search import _tiledbvspy as vspy

        embedding_dim = vectors_array.shape[1]
        num_vectors = vectors_array.shape[0]
        MAX_UINT64 = 18446744073709551615

        logger.info(f"Creating TileDB array: {num_vectors:,} vectors of dimension {embedding_dim}")

        array_uri = str(self.PERSIST_DIRECTORY / "vectors")

        dom = tiledb.Domain(
            tiledb.Dim(name="id", domain=(0, np.iinfo(np.uint64).max - 20000), tile=10000, dtype=np.uint64)
        )

        attrs = [
            tiledb.Attr(name="vector", dtype=np.dtype([("", np.float32)] * embedding_dim)),
            tiledb.Attr(name="text", dtype=str, var=True),
            tiledb.Attr(name="metadata", dtype=str, var=True),
        ]

        schema = tiledb.ArraySchema(
            domain=dom,
            attrs=attrs,
            sparse=True,
            cell_order='row-major',
            tile_order='row-major'
        )

        tiledb.Array.create(array_uri, schema)

        num_batches = (num_vectors + TILEDB_WRITE_BATCH_SIZE - 1) // TILEDB_WRITE_BATCH_SIZE
        logger.info(f"Writing TileDB array in {num_batches} batch(es)")

        all_ids = np.empty(num_vectors, dtype=np.uint64)
        hash_id_mappings = []
        rng = np.random.default_rng()

        for batch_idx in range(num_batches):
            start = batch_idx * TILEDB_WRITE_BATCH_SIZE
            end = min(start + TILEDB_WRITE_BATCH_SIZE, num_vectors)

            # Use numpy's vectorized generator instead of a Python list
            # comprehension over random.randint. The list-comprehension
            # approach allocated end-start Python int objects per batch
            # (~7+ GB total at the Caselaw scale), which triggered an
            # OverflowError + access violation inside random.randint on
            # Python 3.12. numpy's integers() runs entirely in C and
            # returns a uint64 array directly.
            batch_ids = rng.integers(
                low=0,
                high=np.iinfo(np.uint64).max,
                size=end - start,
                dtype=np.uint64,
                endpoint=False,
            )
            all_ids[start:end] = batch_ids

            for i in range(start, end):
                file_hash = metadatas[i].get('hash', '')
                hash_id_mappings.append((str(batch_ids[i - start]), file_hash))

            batch_vectors = vectors_array[start:end]
            batch_texts = np.array(texts[start:end], dtype=object)
            # _json_dumps uses orjson when available (Rust-based, ~10x faster
            # than stdlib json). The stdlib json.dumps loop here triggered an
            # OverflowError + access violation at the Caselaw scale due to
            # heap fragmentation from millions of small string allocations.
            batch_metadata = np.array(
                [_json_dumps(metadatas[i]) for i in range(start, end)],
                dtype=object
            )

            batch_structured = np.array(
                [tuple(vec) for vec in batch_vectors],
                dtype=[("", np.float32)] * embedding_dim
            )

            with tiledb.open(array_uri, mode='w') as A:
                A[batch_ids] = {
                    "vector": batch_structured,
                    "text": batch_texts,
                    "metadata": batch_metadata,
                }

            del batch_structured, batch_texts, batch_metadata, batch_vectors
            gc.collect()

        tiledb.consolidate(array_uri)
        tiledb.vacuum(array_uri)

        index_uri = str(self.PERSIST_DIRECTORY / "vector_index")

        vs.ingest(
            index_type="FLAT",
            index_uri=index_uri,
            input_vectors=vectors_array,
            external_ids=all_ids,
            dimensions=embedding_dim,
            distance_metric=vspy.DistanceMetric.COSINE
        )

        metadata_file = self.PERSIST_DIRECTORY / "index_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'distance_metric': 'cosine',
                'dimensions': embedding_dim,
                'vector_type': 'float32',
                'index_type': 'FLAT',
                'num_vectors': num_vectors
            }, f)

        logger.info(f"FLAT index created at: {index_uri}")
        return hash_id_mappings

    def load_audio_documents(self, source_dir=None):
        if source_dir is None:
            source_dir = self.SOURCE_DIRECTORY
        json_paths = [f for f in source_dir.iterdir() if f.suffix.lower() == '.json']
        docs = []

        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    data = json.loads(json_file.read())
                    doc = Document(
                        page_content=data.get('page_content', ''),
                        metadata=data.get('metadata', {})
                    )
                    docs.append(doc)
            except Exception as e:
                my_cprint(f"Error loading {json_path}: {e}", "red")

        return docs

    def clear_docs_for_db_folder(self):
        for item in self.SOURCE_DIRECTORY.iterdir():
            if item.is_file() or item.is_symlink():
                try:
                    item.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {item}: {e}")

    @torch.inference_mode()
    def run(self):
        cuda_mgr = get_cuda_manager()
        pipeline_t0 = time.time()

        config_data = get_config()
        EMBEDDING_MODEL_NAME = config_data.EMBEDDING_MODEL_NAME
        chunk_size = config_data.database.chunk_size
        chunk_overlap = config_data.database.chunk_overlap

        tmp_dir = tempfile.mkdtemp(prefix="vectordb_create_")
        tmp_path = Path(tmp_dir)
        extracted_pkl = tmp_path / "extracted.pkl"
        chunks_pkl = tmp_path / "chunks.pkl"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        try:
            # Stage 1: Extract documents via subprocess
            my_cprint("Extracting documents (subprocess)...", "yellow")
            extract_t0 = time.time()
            _run_extract_with_retry(self.SOURCE_DIRECTORY, extracted_pkl)
            logger.info(f"Extract stage: {time.time() - extract_t0:.1f}s")

            with open(extracted_pkl, "rb") as f:
                doc_data = pickle.load(f)
            logger.info(f"Extracted {len(doc_data)} documents")

            if not doc_data:
                my_cprint("No documents found to process.", "red")
                return

            json_docs_to_save = []
            for content, metadata in doc_data:
                json_docs_to_save.append(Document(page_content=content, metadata=metadata))

            # Also load audio transcripts and images
            print("Processing any audio transcripts...")
            audio_documents = self.load_audio_documents()
            if audio_documents:
                for doc in audio_documents:
                    doc_data.append((doc.page_content, doc.metadata))
                    json_docs_to_save.append(doc)

            print("Processing any images...")
            try:
                from modules.process_images import choose_image_loader
                image_documents = choose_image_loader()
                if isinstance(image_documents, list) and image_documents:
                    for doc in image_documents:
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                        doc_data.append((content, metadata))
                        json_docs_to_save.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                logger.warning(f"Image processing skipped: {e}")

            # Re-write extracted.pkl with audio+image docs included
            with open(extracted_pkl, "wb") as f:
                pickle.dump(doc_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            del doc_data
            gc.collect()

            # Stage 2: Split documents via subprocess
            my_cprint("Splitting documents into chunks (subprocess)...", "yellow")
            split_t0 = time.time()
            _run_split_with_retry(extracted_pkl, chunks_pkl, chunk_size, chunk_overlap, checkpoint_dir)
            logger.info(f"Split stage: {time.time() - split_t0:.1f}s")

            try:
                extracted_pkl.unlink()
            except Exception:
                pass

            with open(chunks_pkl, "rb") as f:
                split_output = pickle.load(f)

            if isinstance(split_output, dict):
                chunk_texts = split_output["texts"]
                chunks_with_meta = split_output.get("chunks", [])
                del split_output
            else:
                chunk_texts = split_output
                chunks_with_meta = []
                del split_output

            gc.collect()
            logger.info(f"Split into {len(chunk_texts):,} chunks")

            if not chunk_texts:
                my_cprint("No chunks produced after splitting.", "red")
                return

            # Extract metadata dicts from chunks_with_meta, then free it
            all_metadatas = []
            for idx in range(len(chunk_texts)):
                if idx < len(chunks_with_meta):
                    _, meta = chunks_with_meta[idx]
                else:
                    meta = {}
                all_metadatas.append(meta)

            del chunks_with_meta
            gc.collect()

            # Stage 3+4: Tokenize + Embed via subprocess pipeline
            with cuda_mgr.cuda_operation():
                embeddings = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

            my_cprint("\nComputing vectors...", "yellow")
            embed_t0 = time.time()

            try:
                self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=False)
                my_cprint(f"Created directory: {self.PERSIST_DIRECTORY}", "green")
            except FileExistsError:
                raise FileExistsError(
                    f"Vector database '{self.PERSIST_DIRECTORY.name}' already exists. "
                    "Choose a different name or delete the existing DB first."
                )

            with cuda_mgr.cuda_operation():
                vectors = embeddings.embed_documents(chunk_texts)

            embed_elapsed = time.time() - embed_t0
            my_cprint(f"Embedding computation completed in {embed_elapsed:.2f} seconds.", "cyan")

            del embeddings
            gc.collect()
            cuda_mgr.force_empty_cache()

            vectors_array = np.ascontiguousarray(vectors, dtype=np.float32)
            del vectors
            gc.collect()

            # Stage 5: Write TileDB array + FLAT index (IDs generated per-batch)
            try:
                hash_id_mappings = self._create_tiledb_array(chunk_texts, vectors_array, all_metadatas)
            except Exception as e:
                logger.error(f"Error creating TileDB database: {e}")
                traceback.print_exc()
                if self.PERSIST_DIRECTORY.exists():
                    try:
                        shutil.rmtree(self.PERSIST_DIRECTORY)
                    except Exception:
                        pass
                raise

            my_cprint("Processed all chunks", "yellow")

            pipeline_elapsed = time.time() - pipeline_t0
            my_cprint(f"Database created. Total time: {pipeline_elapsed:.2f} seconds.", "green")

            # Stage 6: Write SQLite metadata DB
            del chunk_texts, vectors_array, all_metadatas
            gc.collect()

            create_metadata_db(self.PERSIST_DIRECTORY, json_docs_to_save, hash_id_mappings)
            del json_docs_to_save, hash_id_mappings
            gc.collect()

            self.clear_docs_for_db_folder()

        except Exception:
            traceback.print_exc()
            raise
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


_thread_local = threading.local()


def get_query_db(database_name: str) -> "QueryVectorDB":
    """Return a thread-local QueryVectorDB instance, creating it if needed.

    Each thread gets its own cache of database name → QueryVectorDB, so
    concurrent queries against different databases don't thrash singleton state.
    """
    if not hasattr(_thread_local, "query_db_cache"):
        _thread_local.query_db_cache = {}

    if database_name in _thread_local.query_db_cache:
        return _thread_local.query_db_cache[database_name]

    instance = QueryVectorDB(database_name)
    _thread_local.query_db_cache[database_name] = instance
    return instance


def clear_query_cache(database_name: Optional[str] = None) -> None:
    """Clear the thread-local QueryVectorDB cache for the current thread."""
    if not hasattr(_thread_local, "query_db_cache"):
        return

    if database_name:
        if database_name in _thread_local.query_db_cache:
            _thread_local.query_db_cache[database_name].close()
            del _thread_local.query_db_cache[database_name]
    else:
        for db_instance in _thread_local.query_db_cache.values():
            db_instance.close()
        _thread_local.query_db_cache.clear()


class QueryVectorDB:
    def __init__(self, selected_database: str):
        self.config = self.load_configuration()

        if not selected_database:
            raise ValueError("No vector database selected.")
        if selected_database not in self.config.created_databases:
            raise ValueError(f'Database "{selected_database}" not found in config.')

        db_path = self.config.vector_db_dir / selected_database
        if not db_path.exists():
            raise FileNotFoundError(f'Database folder "{selected_database}" is missing on disk.')

        self.selected_database = selected_database
        self.db_path = db_path
        self.index_uri = str(db_path / "vector_index")
        self.array_uri = str(db_path / "vectors")

        self.embeddings = None
        self.index = None
        self.model_name = None
        self._debug_id = id(self)

        self.distance_metric = "cosine"
        self.index_type = "FLAT"

        try:
            metadata_file = db_path / "index_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.distance_metric = metadata.get('distance_metric', 'cosine')
                    self.index_type = metadata.get('index_type', 'FLAT')
        except Exception as e:
            logger.warning(f"Could not load index metadata, using defaults: {e}")

    def load_configuration(self):
        try:
            return get_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    @torch.inference_mode()
    def initialize_vector_model(self):
        model_path = self.config.created_databases[self.selected_database].model
        self.model_name = os.path.basename(model_path)

        return load_embedding_model(
            model_path=model_path,
            compute_device=self.config.Compute_Device.database_query,
            use_half=self.config.database.half,
            is_query=True,
        )

    @torch.inference_mode()
    def search(self, query, k: Optional[int] = None, score_threshold: Optional[float] = None):
        _setup_tiledb_dlls()
        import tiledb
        import tiledb.vector_search as vs

        cuda_mgr = get_cuda_manager()

        if not self.embeddings:
            logger.info(f"Initializing embedding model for database {self.selected_database}")
            self.embeddings = self.initialize_vector_model()

        if not self.index:
            logger.info(f"Loading TileDB FLAT index for {self.selected_database}")
            self.index = vs.FlatIndex(uri=self.index_uri)

        self.config = self.load_configuration()
        k = k if k is not None else self.config.database.contexts
        score_threshold = score_threshold if score_threshold is not None else self.config.database.similarity

        with cuda_mgr.cuda_operation():
            query_vector = self.embeddings.embed_query(query)

        query_vector_np = np.array([query_vector], dtype=np.float32)

        logger.info(f"Querying TileDB index: {self.index_uri}")

        result_distances, result_ids = self.index.query(query_vector_np, k=k)

        if len(result_distances) == 0 or len(result_distances[0]) == 0:
            logger.warning("No results returned from vector search")
            return [], []

        distances = result_distances[0]
        ids = result_ids[0]

        if len(ids) > 0 and ids[0] == MAX_UINT64_SENTINEL:
            logger.warning("TileDB returned sentinel value - no matches found in index")
            return [], []

        valid_mask = ids != MAX_UINT64_SENTINEL
        distances = distances[valid_mask]
        ids = ids[valid_mask]

        if len(ids) == 0:
            logger.warning("All results were sentinel values - no valid matches")
            return [], []

        logger.info(f"Raw distances - min: {distances.min():.4f}, max: {distances.max():.4f}, mean: {distances.mean():.4f}")

        if self.distance_metric == "cosine":
            similarities = np.clip(1.0 - distances, 0.0, 1.0)
        else:
            logger.warning(f"Unknown distance metric '{self.distance_metric}', assuming cosine")
            similarities = np.clip(1.0 - distances, 0.0, 1.0)

        logger.info(f"Similarities - min: {similarities.min():.4f}, max: {similarities.max():.4f}")
        logger.info(f"Score threshold: {score_threshold}, Results before filtering: {len(similarities)}")

        results = []

        valid_indices = similarities >= score_threshold
        num_passing = np.sum(valid_indices)
        logger.info(f"Results passing threshold: {num_passing}")

        if not np.any(valid_indices):
            logger.warning(f"No results passed the similarity threshold of {score_threshold}")
            return [], []

        filtered_distances = distances[valid_indices]
        filtered_ids = ids[valid_indices]
        filtered_similarities = similarities[valid_indices]

        with tiledb.open(self.array_uri, mode='r') as A:
            data = A.multi_index[filtered_ids.astype(np.uint64)]

            texts_raw = data['text']
            metadatas_raw = data['metadata']

            for i, (distance, vec_id, similarity) in enumerate(zip(filtered_distances, filtered_ids, filtered_similarities)):
                try:
                    text_raw = texts_raw[i]
                    if isinstance(text_raw, np.ndarray):
                        text = text_raw.item() if text_raw.size == 1 else str(text_raw[0])
                    else:
                        text = str(text_raw)

                    metadata_raw = metadatas_raw[i]
                    if isinstance(metadata_raw, np.ndarray):
                        metadata_str = metadata_raw.item() if metadata_raw.size == 1 else str(metadata_raw[0])
                    else:
                        metadata_str = str(metadata_raw)

                    metadata = json.loads(metadata_str)
                    metadata['similarity_score'] = float(similarity)
                    metadata['distance'] = float(distance)
                    results.append((text, metadata))

                except json.JSONDecodeError as je:
                    logger.warning(f"Failed to parse JSON for vector ID {vec_id}: {je}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to retrieve data for vector ID {vec_id}: {e}")
                    continue

        search_term = self.config.database.search_term.lower()
        if search_term:
            filtered_results = [
                (text, metadata) for text, metadata in results
                if search_term in text.lower()
            ]
        else:
            filtered_results = results

        document_types = self.config.database.document_types
        if document_types:
            filtered_results = [
                (text, metadata) for text, metadata in filtered_results
                if metadata.get('document_type') == document_types
            ]

        contexts = [text for text, _ in filtered_results]
        metadata_list = [metadata for _, metadata in filtered_results]

        logger.info(f"Final results returned: {len(contexts)}")
        return contexts, metadata_list

    def cleanup(self):
        if self.embeddings:
            del self.embeddings
            self.embeddings = None

        if self.index:
            del self.index
            self.index = None

        get_cuda_manager().safe_empty_cache()
        gc.collect()

    def close(self):
        self.cleanup()
