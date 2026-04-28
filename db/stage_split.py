import argparse
import concurrent.futures
import gc
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("stage_split")


WORKER_SCRIPT = r'''
import faulthandler
faulthandler.enable()

import gc
import os
import pickle
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.environ.get("VECTORDB_PROJECT_ROOT", ""))

def main():
    input_pkl = sys.argv[1]
    output_pkl = sys.argv[2]
    chunk_size = int(sys.argv[3])
    chunk_overlap = int(sys.argv[4])

    with open(input_pkl, "rb") as f:
        doc_data = pickle.load(f)

    from db.document_processor import FixedSizeTextSplitter, add_pymupdf_page_metadata, Document
    from core.utilities import normalize_text

    splitter = FixedSizeTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks_with_meta = []
    errors = []

    for i, (content, metadata) in enumerate(doc_data):
        try:
            doc = Document(page_content=content, metadata=metadata)

            if metadata.get("file_type") == ".pdf":
                chunks = add_pymupdf_page_metadata(
                    doc,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            else:
                chunks = splitter.split_documents([doc])

            for chunk in chunks:
                cleaned = normalize_text(chunk.page_content, preserve_whitespace=True)
                if cleaned is not None:
                    chunk_meta = chunk.metadata if chunk.metadata else {}
                    chunks_with_meta.append((cleaned, chunk_meta))
        except Exception as e:
            file_name = metadata.get("file_name", "unknown")
            errors.append({
                "doc_index": i,
                "file_name": file_name,
                "error": f"{type(e).__name__}: {e}",
            })

    valid = []
    valid_with_meta = []
    skipped = 0
    for text, meta in chunks_with_meta:
        if isinstance(text, str) and text.strip():
            valid.append(text)
            valid_with_meta.append((text, meta))
        else:
            skipped += 1

    output = {
        "texts": valid,
        "chunks": valid_with_meta,
        "errors": errors,
        "docs_processed": len(doc_data),
        "skipped": skipped,
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
'''


def save_checkpoint(checkpoint_path, data):
    tmp_path = checkpoint_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        tmp_path.rename(checkpoint_path)
    except Exception:
        try:
            tmp_path.unlink()
        except Exception:
            pass
        with open(checkpoint_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_worker(python_exe: str, worker_script_path: str,
               docs_pkl: str, output_pkl: str,
               chunk_size: int, chunk_overlap: int,
               timeout: int = 600) -> tuple:
    cmd = [
        python_exe, worker_script_path,
        docs_pkl, output_pkl,
        str(chunk_size), str(chunk_overlap),
    ]

    project_root = str(Path(__file__).resolve().parent.parent)

    t0 = time.time()
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "VECTORDB_PROJECT_ROOT": project_root},
    ) as process:
        output_lines = []
        for line in process.stdout:
            line = line.rstrip("\n")
            if line.strip():
                logger.warning(f"  [worker] {line}")
                output_lines.append(line)

        process.wait(timeout=timeout)
        elapsed = time.time() - t0
        returncode = process.returncode
    return returncode, elapsed


def get_physical_core_count() -> int:
    try:
        import psutil
        count = psutil.cpu_count(logical=False)
        if count is not None and count > 0:
            return count
    except ImportError:
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2)


def run_worker_with_retries(worker_id: int, total_workers: int,
                            python_exe: str, worker_script_path: str,
                            chunk_docs: list, worker_dir: Path,
                            chunk_size: int, chunk_overlap: int,
                            max_retries: int) -> dict:
    num_docs = len(chunk_docs)
    docs_pkl = worker_dir / f"_split_worker_input_{worker_id}.pkl"
    result_pkl = worker_dir / f"_split_worker_output_{worker_id}.pkl"

    with open(docs_pkl, "wb") as f:
        pickle.dump(chunk_docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    worker_t0 = time.time()
    worker_success = False
    texts = []
    chunks = []
    errors = []
    skipped = 0

    for retry in range(max_retries):
        exit_code, elapsed = run_worker(
            python_exe, str(worker_script_path),
            str(docs_pkl), str(result_pkl),
            chunk_size, chunk_overlap,
            timeout=600,
        )

        if exit_code == 0 and result_pkl.exists():
            try:
                with open(result_pkl, "rb") as f:
                    worker_data = pickle.load(f)
                texts = worker_data.get("texts", [])
                chunks = worker_data.get("chunks", [])
                errors = worker_data.get("errors", [])
                skipped = worker_data.get("skipped", 0)
                worker_success = True
                logger.info(f"  Worker {worker_id}/{total_workers} completed in {elapsed:.1f}s "
                           f"({len(texts)} chunks, {len(errors)} errors, {skipped} skipped)")
                break
            except Exception as e:
                logger.error(f"  Worker {worker_id}: failed to read output: {e}")
        else:
            logger.warning(f"  Worker {worker_id} crashed (exit code {exit_code}, "
                         f"{elapsed:.1f}s), retry {retry + 1}/{max_retries}")
            time.sleep(2)

        try:
            if result_pkl.exists():
                result_pkl.unlink()
        except Exception:
            pass

    try:
        docs_pkl.unlink()
    except Exception:
        pass
    try:
        if result_pkl.exists():
            result_pkl.unlink()
    except Exception:
        pass

    if not worker_success:
        logger.error(f"  Worker {worker_id} FAILED after {max_retries} retries, "
                    f"skipping {num_docs} documents")
        errors.append({
            "doc_index": -1,
            "file_name": "BATCH_FAILURE",
            "error": f"Worker crashed {max_retries} times",
        })

    worker_elapsed = time.time() - worker_t0
    return {
        "worker_id": worker_id,
        "num_docs": num_docs,
        "success": worker_success,
        "texts": texts,
        "chunks": chunks,
        "errors": errors,
        "skipped": skipped,
        "elapsed": worker_elapsed,
    }


def main():
    import faulthandler
    faulthandler.enable()

    parser = argparse.ArgumentParser(description="Stage 2: Text Splitting (subprocess-per-chunk)")
    parser.add_argument("input_pickle", type=Path)
    parser.add_argument("output_pickle", type=Path)
    parser.add_argument("chunk_size", type=int)
    parser.add_argument("chunk_overlap", type=int)
    parser.add_argument("--worker-batch-size", type=int, default=2000)
    parser.add_argument("--max-worker-retries", type=int, default=3)
    parser.add_argument("--max-parallel-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    args = parser.parse_args()

    if not args.input_pickle.exists():
        print(f"ERROR: Input file does not exist: {args.input_pickle}", file=sys.stderr)
        sys.exit(1)

    python_exe = sys.executable
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "split_checkpoint.pkl"

    worker_dir = checkpoint_dir if checkpoint_dir else Path(tempfile.gettempdir())
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_script_path = worker_dir / "_split_worker.py"
    with open(worker_script_path, "w", encoding="utf-8") as f:
        f.write(WORKER_SCRIPT)

    with open(args.input_pickle, "rb") as f:
        doc_data = pickle.load(f)

    total_docs = len(doc_data)

    MIN_DOCS_FOR_PARALLEL = 5000
    if args.max_parallel_workers > 0:
        max_parallel = args.max_parallel_workers
    else:
        physical_cores = get_physical_core_count()
        max_parallel = max(1, physical_cores - 2)

    if total_docs <= MIN_DOCS_FOR_PARALLEL:
        effective_parallel = 1
    else:
        effective_parallel = max_parallel

    logger.info(f"Stage 2: Splitting documents (subprocess-per-chunk isolation)")
    logger.info(f"  chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}")
    logger.info(f"  worker_batch_size={args.worker_batch_size}")
    logger.info(f"  parallel_workers={effective_parallel}")
    t0 = time.time()

    logger.info(f"Loaded {total_docs} documents from {args.input_pickle}")

    if total_docs == 0:
        logger.info("No documents to process")
        with open(args.output_pickle, "wb") as f:
            pickle.dump({"texts": [], "chunks": []}, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    worker_batch_size = args.worker_batch_size
    worker_jobs = []
    offset = 0
    worker_id = 0
    while offset < total_docs:
        worker_id += 1
        chunk_end = min(offset + worker_batch_size, total_docs)
        chunk_docs = doc_data[offset:chunk_end]
        worker_jobs.append((worker_id, chunk_docs))
        offset = chunk_end

    total_workers = len(worker_jobs)
    logger.info(f"Processing {total_docs} documents in {total_workers} worker subprocess(es)")

    all_texts = []
    all_chunks = []
    all_errors = []
    total_skipped = 0
    workers_completed = 0
    workers_since_checkpoint = 0

    if effective_parallel <= 1:
        for wid, chunk_docs in worker_jobs:
            logger.info(f"Worker {wid}/{total_workers}: {len(chunk_docs)} documents")
            result = run_worker_with_retries(
                wid, total_workers, python_exe, str(worker_script_path),
                chunk_docs, worker_dir, args.chunk_size, args.chunk_overlap,
                args.max_worker_retries,
            )
            all_texts.extend(result["texts"])
            all_chunks.extend(result.get("chunks", []))
            all_errors.extend(result["errors"])
            total_skipped += result["skipped"]
            workers_completed += 1
            workers_since_checkpoint += 1

            if checkpoint_path is not None and workers_since_checkpoint >= args.checkpoint_interval:
                save_checkpoint(checkpoint_path, {
                    "texts": all_texts, "chunks": all_chunks,
                    "errors": all_errors, "skipped": total_skipped,
                    "workers_completed": workers_completed,
                })
                workers_since_checkpoint = 0

            gc.collect()
    else:
        for wave_start in range(0, total_workers, effective_parallel):
            wave_end = min(wave_start + effective_parallel, total_workers)
            wave_jobs = worker_jobs[wave_start:wave_end]

            logger.info(f"Launching parallel wave: workers {wave_jobs[0][0]}-{wave_jobs[-1][0]}")

            wave_results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave_jobs)) as executor:
                future_to_wid = {}
                for wid, chunk_docs in wave_jobs:
                    future = executor.submit(
                        run_worker_with_retries,
                        wid, total_workers, python_exe, str(worker_script_path),
                        chunk_docs, worker_dir, args.chunk_size, args.chunk_overlap,
                        args.max_worker_retries,
                    )
                    future_to_wid[future] = wid

                for future in concurrent.futures.as_completed(future_to_wid):
                    wid = future_to_wid[future]
                    try:
                        result = future.result()
                        wave_results[wid] = result
                    except Exception as e:
                        logger.error(f"  Worker {wid} thread raised exception: {e}")
                        wave_results[wid] = {
                            "texts": [], "chunks": [], "errors": [{"doc_index": -1,
                            "file_name": "THREAD_EXCEPTION", "error": str(e)}], "skipped": 0,
                        }

            for wid, chunk_docs in wave_jobs:
                result = wave_results.get(wid, {"texts": [], "chunks": [], "errors": [], "skipped": 0})
                all_texts.extend(result["texts"])
                all_chunks.extend(result.get("chunks", []))
                all_errors.extend(result["errors"])
                total_skipped += result.get("skipped", 0)
                workers_completed += 1
                workers_since_checkpoint += 1

            if checkpoint_path is not None and workers_since_checkpoint >= args.checkpoint_interval:
                save_checkpoint(checkpoint_path, {
                    "texts": all_texts, "chunks": all_chunks,
                    "errors": all_errors, "skipped": total_skipped,
                    "workers_completed": workers_completed,
                })
                workers_since_checkpoint = 0

            gc.collect()

    elapsed = time.time() - t0
    logger.info(f"Split {total_docs} documents into {len(all_texts)} chunks in {elapsed:.1f}s "
                f"({len(all_errors)} errors, {total_skipped} skipped)")

    output_data = {"texts": all_texts, "chunks": all_chunks}
    with open(args.output_pickle, "wb") as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        worker_script_path.unlink()
    except Exception:
        pass
    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
