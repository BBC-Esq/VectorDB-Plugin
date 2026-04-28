import argparse
import concurrent.futures
import gc
import hashlib
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
logger = logging.getLogger("stage_tokenize")


WORKER_SCRIPT = r'''
import faulthandler
faulthandler.enable()

import gc
import os
import pickle
import sys

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RUST_BACKTRACE"] = "1"

def main():
    input_pkl = sys.argv[1]
    output_pkl = sys.argv[2]
    model_path = sys.argv[3]
    batch_size = int(sys.argv[4])
    max_seq_length = int(sys.argv[5])
    use_fast = sys.argv[6] == "True"
    global_start_index = int(sys.argv[7])
    encode_batch_size = int(sys.argv[8])
    length_sort = sys.argv[9] == "True"

    with open(input_pkl, "rb") as f:
        texts = pickle.load(f)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        model_max_length=max_seq_length,
        use_fast=use_fast,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    pad_token_id = tokenizer.pad_token_id or 0
    padding_side = getattr(tokenizer, "padding_side", "right")

    all_sequences = []
    errors_result = []

    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_id = start // batch_size + 1
        global_start = global_start_index + start

        try:
            batch_raw = tokenizer(
                batch_texts,
                padding=False,
                truncation=True,
                max_length=max_seq_length,
                return_tensors=None,
                return_attention_mask=True,
            )
            num_texts_in_batch = len(batch_raw["input_ids"])
            keys = list(batch_raw.keys())
            for i in range(num_texts_in_batch):
                seq_dict = {"seq_index": global_start + i}
                for key in keys:
                    val = batch_raw[key][i]
                    if not isinstance(val, list):
                        seq_dict[key] = list(val)
                    else:
                        seq_dict[key] = val
                all_sequences.append(seq_dict)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            errors_result.append({
                "batch_id": batch_id,
                "start_index": global_start,
                "error": error_msg,
            })

    if length_sort and all_sequences:
        all_sequences.sort(key=lambda s: len(s["input_ids"]), reverse=True)

    feature_keys = [k for k in all_sequences[0].keys() if k != "seq_index"] if all_sequences else []

    batches_result = []
    total_real_tokens = 0
    total_pad_tokens = 0

    for b_start in range(0, len(all_sequences), encode_batch_size):
        b_end = min(b_start + encode_batch_size, len(all_sequences))
        batch_seqs = all_sequences[b_start:b_end]
        batch_id = b_start // encode_batch_size + 1
        start_index = batch_seqs[0]["seq_index"]
        batch_size_actual = len(batch_seqs)
        max_len = max(len(s["input_ids"]) for s in batch_seqs)

        result = {}
        for key in feature_keys:
            pad_val = pad_token_id if key == "input_ids" else 0
            padded = np.full((batch_size_actual, max_len), pad_val, dtype=np.int64)

            for i, seq in enumerate(batch_seqs):
                seq_data = seq[key]
                seq_len = len(seq_data)
                if padding_side == "left":
                    padded[i, max_len - seq_len:] = seq_data
                else:
                    padded[i, :seq_len] = seq_data

                if key == "input_ids":
                    total_real_tokens += seq_len
                    total_pad_tokens += (max_len - seq_len)

            result[key] = padded

        seq_indices = np.array([s["seq_index"] for s in batch_seqs], dtype=np.int64)

        batches_result.append({
            "batch_id": batch_id,
            "start_index": start_index,
            "seq_indices": seq_indices,
            "features": result,
        })

    del all_sequences
    gc.collect()

    total_tokens = total_real_tokens + total_pad_tokens
    efficiency_pct = (total_real_tokens / total_tokens * 100) if total_tokens > 0 else 100.0

    output = {
        "batches": batches_result,
        "errors": errors_result,
        "texts_processed": len(texts),
        "padding_stats": {
            "total_real_tokens": total_real_tokens,
            "total_pad_tokens": total_pad_tokens,
            "efficiency_pct": efficiency_pct,
        },
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
'''


def save_checkpoint(checkpoint_path, data, fingerprint):
    payload = {**data, "fingerprint": fingerprint}
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    for attempt in range(5):
        try:
            os.replace(tmp_path, checkpoint_path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2)


def _compute_input_hash(input_path):
    h = hashlib.sha256()
    with open(input_path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def load_checkpoint(checkpoint_path, expected_fingerprint):
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint at {checkpoint_path}: {e}; starting fresh")
        return None
    if data.get("fingerprint") != expected_fingerprint:
        logger.warning(
            f"Checkpoint at {checkpoint_path} has a fingerprint mismatch "
            f"(input or build args changed since the checkpoint was written); starting fresh"
        )
        return None
    return data


def run_worker(python_exe: str, worker_script_path: str,
               texts_pkl: str, output_pkl: str,
               model_path: str, batch_size: int, max_seq_length: int,
               use_fast: bool, global_start_index: int,
               encode_batch_size: int, length_sort: bool,
               timeout: int = 600) -> tuple:
    cmd = [
        python_exe, worker_script_path,
        texts_pkl, output_pkl,
        model_path, str(batch_size), str(max_seq_length),
        str(use_fast), str(global_start_index),
        str(encode_batch_size), str(length_sort),
    ]

    t0 = time.time()
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
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


def run_worker_with_retries(worker_id, total_workers, python_exe, worker_script_path,
                            chunk_texts, global_start, worker_dir, model_path,
                            batch_size, max_seq_length, use_fast, max_retries,
                            encode_batch_size, length_sort) -> dict:
    num_texts = len(chunk_texts)
    chunk_pkl = worker_dir / f"_worker_input_{worker_id}.pkl"
    result_pkl = worker_dir / f"_worker_output_{worker_id}.pkl"

    with open(chunk_pkl, "wb") as f:
        pickle.dump(chunk_texts, f, protocol=pickle.HIGHEST_PROTOCOL)

    worker_t0 = time.time()
    worker_success = False
    batches = []
    errors = []
    padding_stats = {}

    for retry in range(max_retries):
        exit_code, elapsed = run_worker(
            python_exe, str(worker_script_path),
            str(chunk_pkl), str(result_pkl),
            model_path, batch_size, max_seq_length,
            use_fast, global_start,
            encode_batch_size, length_sort,
            timeout=600,
        )

        if exit_code == 0 and result_pkl.exists():
            try:
                with open(result_pkl, "rb") as f:
                    worker_data = pickle.load(f)
                batches = worker_data.get("batches", [])
                errors = worker_data.get("errors", [])
                padding_stats = worker_data.get("padding_stats", {})
                worker_success = True
                eff = padding_stats.get("efficiency_pct", 0)
                logger.info(f"  Worker {worker_id}/{total_workers} completed in {elapsed:.1f}s "
                           f"({len(batches)} batches, {eff:.1f}% pad efficiency)")
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
        chunk_pkl.unlink()
    except Exception:
        pass
    try:
        if result_pkl.exists():
            result_pkl.unlink()
    except Exception:
        pass

    if not worker_success:
        logger.error(f"  Worker {worker_id} FAILED after {max_retries} retries")
        for batch_start in range(0, num_texts, batch_size):
            errors.append({
                "batch_id": -1,
                "start_index": global_start + batch_start,
                "error": f"Worker crashed {max_retries} times",
            })

    worker_elapsed = time.time() - worker_t0
    return {
        "worker_id": worker_id,
        "global_start": global_start,
        "num_texts": num_texts,
        "success": worker_success,
        "batches": batches,
        "errors": errors,
        "padding_stats": padding_stats,
        "elapsed": worker_elapsed,
    }


def main():
    import faulthandler
    faulthandler.enable()

    parser = argparse.ArgumentParser(description="Stage 3: Tokenization (subprocess-per-chunk)")
    parser.add_argument("input_pickle", type=Path)
    parser.add_argument("output_pickle", type=Path)
    parser.add_argument("model_path", type=str)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("max_seq_length", type=int)
    parser.add_argument("--use-fast", action="store_true", default=True)
    parser.add_argument("--no-use-fast", dest="use_fast", action="store_false")
    parser.add_argument("--worker-batch-size", type=int, default=20000)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--start-text-index", type=int, default=0)
    parser.add_argument("--max-worker-retries", type=int, default=3)
    parser.add_argument("--max-parallel-workers", type=int, default=0)
    parser.add_argument("--encode-batch-size", type=int, required=True)
    parser.add_argument("--length-sort", action="store_true", default=False)
    parser.add_argument("--no-length-sort", dest="length_sort", action="store_false")
    args = parser.parse_args()

    if not args.input_pickle.exists():
        print(f"ERROR: Input file does not exist: {args.input_pickle}", file=sys.stderr)
        sys.exit(1)

    python_exe = sys.executable
    start_text_index = args.start_text_index
    checkpoint_dir = args.checkpoint_dir
    checkpoint_interval = args.checkpoint_interval
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "tokenize_checkpoint.pkl"

    worker_dir = checkpoint_dir if checkpoint_dir else Path(tempfile.gettempdir())
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_script_path = worker_dir / "_tokenize_worker.py"
    with open(worker_script_path, "w", encoding="utf-8") as f:
        f.write(WORKER_SCRIPT)

    with open(args.input_pickle, "rb") as f:
        _loaded = pickle.load(f)
    if isinstance(_loaded, dict):
        all_texts = _loaded["texts"]
    else:
        all_texts = _loaded
    del _loaded

    total_all = len(all_texts)

    if start_text_index > 0:
        logger.info(f"Resuming from text index {start_text_index}")

    texts = all_texts[start_text_index:]
    total = len(texts)

    MIN_TEXTS_FOR_PARALLEL = 5000
    if args.max_parallel_workers > 0:
        max_parallel = args.max_parallel_workers
    else:
        physical_cores = get_physical_core_count()
        max_parallel = max(1, physical_cores - 4)

    if total <= MIN_TEXTS_FOR_PARALLEL:
        effective_parallel = 1
    else:
        effective_parallel = max_parallel

    logger.info(f"Stage 3: Tokenizing (subprocess-per-chunk isolation)")
    logger.info(f"  batch_size={args.batch_size}, max_seq_length={args.max_seq_length}")
    logger.info(f"  encode_batch_size={args.encode_batch_size}, length_sort={args.length_sort}")
    logger.info(f"  parallel_workers={effective_parallel}")
    t0 = time.time()

    if total == 0:
        logger.info("No texts to process")
        output = {
            "total_texts": total_all, "batch_size": args.batch_size,
            "encode_batch_size": args.encode_batch_size,
            "start_text_index": start_text_index, "texts_processed": 0,
            "batches": [], "errors": [],
            "padding_stats": {"total_real_tokens": 0, "total_pad_tokens": 0, "efficiency_pct": 100.0},
        }
        with open(args.output_pickle, "wb") as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    worker_batch_size = args.worker_batch_size
    worker_jobs = []
    offset = 0
    worker_id = 0
    while offset < total:
        worker_id += 1
        chunk_end = min(offset + worker_batch_size, total)
        chunk_texts = texts[offset:chunk_end]
        global_start = start_text_index + offset
        worker_jobs.append((worker_id, global_start, chunk_texts))
        offset = chunk_end

    total_workers = len(worker_jobs)
    logger.info(f"Processing {total} texts in {total_workers} worker subprocess(es)")

    fingerprint = None
    ckpt = None
    if checkpoint_path is not None:
        fingerprint = {
            "stage": "tokenize",
            "input_hash": _compute_input_hash(args.input_pickle),
            "model_path": args.model_path,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "encode_batch_size": args.encode_batch_size,
            "length_sort": args.length_sort,
            "use_fast": args.use_fast,
            "start_text_index": start_text_index,
            "worker_batch_size": worker_batch_size,
            "total_workers": total_workers,
        }
        ckpt = load_checkpoint(checkpoint_path, fingerprint)

    if ckpt is not None:
        all_batches = ckpt["batches"]
        all_errors = ckpt["errors"]
        ps = ckpt.get("padding_stats", {})
        total_real_tokens = ps.get("total_real_tokens", 0)
        total_pad_tokens = ps.get("total_pad_tokens", 0)
        workers_completed = ckpt["workers_completed"]
        logger.info(
            f"Resuming from checkpoint: {workers_completed}/{total_workers} workers already complete, "
            f"{len(all_batches)} batches accumulated"
        )
    else:
        all_batches = []
        all_errors = []
        total_real_tokens = 0
        total_pad_tokens = 0
        workers_completed = 0
    workers_since_checkpoint = 0

    if effective_parallel <= 1:
        for wid, gstart, chunk in worker_jobs:
            if wid <= workers_completed:
                continue
            logger.info(f"Worker {wid}/{total_workers}: {len(chunk)} texts")
            result = run_worker_with_retries(
                wid, total_workers, python_exe, str(worker_script_path),
                chunk, gstart, worker_dir, args.model_path,
                args.batch_size, args.max_seq_length, args.use_fast,
                args.max_worker_retries,
                args.encode_batch_size, args.length_sort,
            )
            all_batches.extend(result["batches"])
            all_errors.extend(result["errors"])
            ps = result.get("padding_stats", {})
            total_real_tokens += ps.get("total_real_tokens", 0)
            total_pad_tokens += ps.get("total_pad_tokens", 0)
            workers_completed += 1
            workers_since_checkpoint += 1

            if checkpoint_path is not None and workers_since_checkpoint >= checkpoint_interval:
                current_offset = gstart + result["num_texts"] - start_text_index
                save_checkpoint(checkpoint_path, {
                    "total_texts": total_all, "start_text_index": start_text_index,
                    "texts_processed": current_offset,
                    "batches": all_batches, "errors": all_errors,
                    "next_text_index": start_text_index + current_offset,
                    "padding_stats": {"total_real_tokens": total_real_tokens, "total_pad_tokens": total_pad_tokens},
                    "workers_completed": workers_completed,
                }, fingerprint)
                workers_since_checkpoint = 0

            gc.collect()
    else:
        for wave_start in range(0, total_workers, effective_parallel):
            wave_end = min(wave_start + effective_parallel, total_workers)
            wave_jobs = worker_jobs[wave_start:wave_end]
            wave_jobs = [(wid, gstart, chunk) for wid, gstart, chunk in wave_jobs if wid > workers_completed]
            if not wave_jobs:
                continue

            logger.info(f"Launching parallel wave: workers {wave_jobs[0][0]}-{wave_jobs[-1][0]}")

            wave_results = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave_jobs)) as executor:
                future_to_wid = {}
                for wid, gstart, chunk in wave_jobs:
                    future = executor.submit(
                        run_worker_with_retries,
                        wid, total_workers, python_exe, str(worker_script_path),
                        chunk, gstart, worker_dir, args.model_path,
                        args.batch_size, args.max_seq_length, args.use_fast,
                        args.max_worker_retries,
                        args.encode_batch_size, args.length_sort,
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
                            "batches": [], "errors": [{"batch_id": -1, "start_index": -1,
                            "error": str(e)}], "padding_stats": {},
                        }

            for wid, gstart, chunk in wave_jobs:
                result = wave_results.get(wid, {"batches": [], "errors": [], "num_texts": len(chunk), "padding_stats": {}})
                all_batches.extend(result["batches"])
                all_errors.extend(result["errors"])
                ps = result.get("padding_stats", {})
                total_real_tokens += ps.get("total_real_tokens", 0)
                total_pad_tokens += ps.get("total_pad_tokens", 0)
                workers_completed += 1
                workers_since_checkpoint += 1

            if checkpoint_path is not None and workers_since_checkpoint >= checkpoint_interval:
                last_wid, last_gstart, last_chunk = wave_jobs[-1]
                current_offset = last_gstart + len(last_chunk) - start_text_index
                save_checkpoint(checkpoint_path, {
                    "total_texts": total_all, "start_text_index": start_text_index,
                    "texts_processed": current_offset,
                    "batches": all_batches, "errors": all_errors,
                    "next_text_index": start_text_index + current_offset,
                    "padding_stats": {"total_real_tokens": total_real_tokens, "total_pad_tokens": total_pad_tokens},
                    "workers_completed": workers_completed,
                }, fingerprint)
                workers_since_checkpoint = 0

            gc.collect()

    elapsed = time.time() - t0

    total_tokens = total_real_tokens + total_pad_tokens
    efficiency_pct = (total_real_tokens / total_tokens * 100) if total_tokens > 0 else 100.0

    logger.info(f"Tokenized {len(all_batches)} batches in {elapsed:.1f}s "
                f"({len(all_errors)} errors, {efficiency_pct:.1f}% pad efficiency)")

    output = {
        "total_texts": total_all,
        "batch_size": args.batch_size,
        "encode_batch_size": args.encode_batch_size,
        "start_text_index": start_text_index,
        "texts_processed": total,
        "batches": all_batches,
        "errors": all_errors,
        "padding_stats": {
            "total_real_tokens": total_real_tokens,
            "total_pad_tokens": total_pad_tokens,
            "efficiency_pct": efficiency_pct,
        },
    }

    with open(args.output_pickle, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

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
