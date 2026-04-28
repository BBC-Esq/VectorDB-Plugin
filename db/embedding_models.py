import gc
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

from core.config import get_config
from core.utilities import (
    supports_flash_attention,
    get_embedding_dtype_and_batch,
    get_model_native_precision,
)

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_TOKENIZE_PATH = PROJECT_ROOT / "db" / "stage_tokenize.py"

from core.constants import PIPELINE_PRESETS

TOKENIZE_BATCH_SIZE = 100
WORKER_BATCH_SIZE = 60000
MAX_WORKER_RETRIES = 3
TOKENIZE_MAX_RETRIES = 5
TOKENIZE_CHECKPOINT_INTERVAL = 5


def _get_tokenize_parallel_workers():
    try:
        preset_name = get_config().database.pipeline_preset
    except Exception:
        preset_name = "normal"
    preset = PIPELINE_PRESETS.get(preset_name, PIPELINE_PRESETS["normal"])
    return preset["tokenize_max_parallel_workers"]


def _get_model_family(model_path: str) -> str:
    model_path_lower = model_path.lower()
    if "qwen" in model_path_lower or "qwen3-embedding" in model_path_lower:
        return "qwen"
    if "bge" in model_path_lower:
        return "bge"
    return "generic"


def _get_prompt_for_family(family: str, is_query: bool = False) -> str:
    if family == "qwen" and is_query:
        return "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    if family == "bge" and is_query:
        return "Represent this sentence for searching relevant passages: "
    return ""


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    cleaned = []
    for char in text:
        if char in "\n\t\r":
            cleaned.append(" ")
        elif ord(char) < 32:
            continue
        elif ord(char) == 127:
            continue
        elif ord(char) > 65535:
            continue
        else:
            cleaned.append(char)

    result = "".join(cleaned)
    result = " ".join(result.split())

    return result.strip() or " "


ENCODE_BATCH_SIZE_BY_MODEL = {
    "bge-small-en-v1.5": 100,
    "bge-base-en-v1.5": 80,
    "bge-large-en-v1.5": 50,
    "Qwen3-Embedding-0.6B": 10,
    "Qwen3-Embedding-4B": 5,
}


def _get_encode_batch_size(device: str, model_path: str = "") -> int:
    model_name = os.path.basename(model_path).lower() if model_path else ""
    for key, batch_size in ENCODE_BATCH_SIZE_BY_MODEL.items():
        if key.lower() in model_name:
            logger.info(f"  ENCODE_BATCH_SIZE: {batch_size} (model-aware default for {key})")
            return batch_size

    if device.startswith("cuda"):
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024 ** 3)
            batch_size = max(10, min(256, int(vram_gb * 4)))
            logger.info(f"  ENCODE_BATCH_SIZE: {batch_size} (VRAM fallback, "
                        f"GPU: {gpu_props.name}, {vram_gb:.1f} GB)")
            return batch_size
        except Exception as e:
            logger.warning(f"  Could not query GPU: {e}, defaulting to 10")
            return 10
    else:
        logger.info(f"  ENCODE_BATCH_SIZE: 10 (CPU mode)")
        return 10


def _run_subprocess_stage(name, cmd, cwd, timeout=3600):
    logger.info(f"Starting subprocess stage: {name}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(cwd),
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


def _run_tokenize_with_retry(
    python_exe, model_path, texts_pkl, tokenized_pkl,
    checkpoint_dir, max_seq_length, encode_batch_size,
    use_fast=True, length_sort=True,
):
    all_batches = []
    all_errors = []
    total_real_tokens = 0
    total_pad_tokens = 0
    current_start_index = 0
    total_texts = None
    attempt = 0

    while attempt < TOKENIZE_MAX_RETRIES:
        attempt += 1

        attempt_output = checkpoint_dir / f"tokenized_attempt_{attempt}.pkl"

        logger.info(f"Tokenize attempt {attempt}/{TOKENIZE_MAX_RETRIES} "
                    f"(starting from text index {current_start_index})")

        tokenize_cmd = [
            python_exe, str(STAGE_TOKENIZE_PATH),
            str(texts_pkl),
            str(attempt_output),
            model_path,
            str(TOKENIZE_BATCH_SIZE),
            str(max_seq_length),
            "--checkpoint-dir", str(checkpoint_dir),
            "--checkpoint-interval", str(TOKENIZE_CHECKPOINT_INTERVAL),
            "--start-text-index", str(current_start_index),
            "--worker-batch-size", str(WORKER_BATCH_SIZE),
            "--max-worker-retries", str(MAX_WORKER_RETRIES),
            "--max-parallel-workers", str(_get_tokenize_parallel_workers()),
            "--encode-batch-size", str(encode_batch_size),
        ]
        if use_fast:
            tokenize_cmd.append("--use-fast")
        else:
            tokenize_cmd.append("--no-use-fast")
        if length_sort:
            tokenize_cmd.append("--length-sort")
        else:
            tokenize_cmd.append("--no-length-sort")

        exit_code, _ = _run_subprocess_stage(
            f"Tokenize (attempt {attempt})", tokenize_cmd, cwd=PROJECT_ROOT)

        attempt_data = None
        checkpoint_path = checkpoint_dir / "tokenize_checkpoint.pkl"

        if exit_code == 0 and attempt_output.exists():
            logger.info(f"Attempt {attempt} completed successfully")
            with open(attempt_output, "rb") as f:
                attempt_data = pickle.load(f)
            try:
                attempt_output.unlink()
            except Exception:
                pass

        elif checkpoint_path.exists():
            logger.warning(f"Attempt {attempt} crashed (exit code {exit_code}), "
                           f"loading checkpoint...")
            try:
                with open(checkpoint_path, "rb") as f:
                    attempt_data = pickle.load(f)
                try:
                    checkpoint_path.unlink()
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Failed to read checkpoint: {e}")

            if attempt_output.exists():
                try:
                    attempt_output.unlink()
                except Exception:
                    pass

        else:
            logger.error(f"Attempt {attempt} crashed with no recoverable data")

        if attempt_data is not None:
            if total_texts is None:
                total_texts = attempt_data.get("total_texts", 0)

            new_batches = attempt_data.get("batches", [])
            new_errors = attempt_data.get("errors", [])
            texts_processed = attempt_data.get("texts_processed", 0)

            all_batches.extend(new_batches)
            all_errors.extend(new_errors)

            ps = attempt_data.get("padding_stats", {})
            total_real_tokens += ps.get("total_real_tokens", 0)
            total_pad_tokens += ps.get("total_pad_tokens", 0)

            if "next_text_index" in attempt_data:
                next_index = attempt_data["next_text_index"]
            else:
                next_index = attempt_data.get("start_text_index", current_start_index) + texts_processed

            current_start_index = next_index

            if total_texts is not None and current_start_index >= total_texts:
                break

            if exit_code == 0:
                break
        else:
            logger.warning(f"No data recovered from attempt {attempt}")

        if attempt >= TOKENIZE_MAX_RETRIES:
            logger.error(f"Exhausted all {TOKENIZE_MAX_RETRIES} retries!")
            break

        logger.info("Waiting 3 seconds before retry...")
        time.sleep(3)
        gc.collect()

    total_tokens = total_real_tokens + total_pad_tokens
    efficiency_pct = (total_real_tokens / total_tokens * 100) if total_tokens > 0 else 100.0

    logger.info(f"Tokenization complete: {len(all_batches)} batches, "
                f"{len(all_errors)} errors, {efficiency_pct:.1f}% padding efficiency")

    return {
        "total_texts": total_texts or 0,
        "batches": all_batches,
        "errors": all_errors,
        "padding_stats": {
            "total_real_tokens": total_real_tokens,
            "total_pad_tokens": total_pad_tokens,
            "efficiency_pct": efficiency_pct,
        },
    }


class DirectEmbeddingModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = None,
        batch_size: int = 8,
        max_seq_length: int = 512,
        prompt: str = "",
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.prompt = prompt
        self.model = None
        self.tokenizer = None

        self._initialize_model()

    def _initialize_model(self):
        family = _get_model_family(self.model_path)

        model_kwargs = {
            "torch_dtype": self.dtype if self.dtype else torch.float32,
        }

        is_cuda = self.device.lower().startswith("cuda")
        if family == "qwen":
            if is_cuda and supports_flash_attention():
                model_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                model_kwargs["attn_implementation"] = "sdpa"
        else:
            model_kwargs["attn_implementation"] = "sdpa"

        tokenizer_kwargs = {
            "model_max_length": self.max_seq_length,
        }

        if family == "qwen":
            tokenizer_kwargs["padding_side"] = "left"

        self.model = SentenceTransformer(
            model_name_or_path=self.model_path,
            device=self.device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        self.model.max_seq_length = self.max_seq_length

        if hasattr(self.model, "tokenizer") and self.model.tokenizer is not None:
            self.tokenizer = self.model.tokenizer

            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.model.to(self.device)

    def _safe_encode(self, texts: list) -> np.ndarray:
        bs = self.batch_size if self.batch_size else len(texts)
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(embeddings, torch.Tensor):
            return embeddings.float().cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)

    @torch.inference_mode()
    def embed_documents(self, texts: list) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        total = len(texts)
        logger.info(f"Embedding {total} texts via subprocess tokenization pipeline")

        encode_batch_size = _get_encode_batch_size(self.device, self.model_path)

        tmp_dir = tempfile.mkdtemp(prefix="vectordb_embed_")
        tmp_path = Path(tmp_dir)
        texts_pkl = tmp_path / "texts.pkl"
        tokenized_pkl = tmp_path / "tokenized.pkl"
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        try:
            logger.info(f"Writing {total} texts to temp pickle...")
            with open(texts_pkl, "wb") as f:
                pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)

            tokenized_data = _run_tokenize_with_retry(
                python_exe=sys.executable,
                model_path=self.model_path,
                texts_pkl=texts_pkl,
                tokenized_pkl=tokenized_pkl,
                checkpoint_dir=checkpoint_dir,
                max_seq_length=self.max_seq_length,
                encode_batch_size=encode_batch_size,
                use_fast=True,
                length_sort=True,
            )

            batches = tokenized_data["batches"]
            errors = tokenized_data["errors"]

            if errors:
                logger.warning(f"{len(errors)} tokenization errors occurred")

            logger.info(f"Running forward pass on {len(batches)} pre-padded batches...")

            self.model.eval()
            all_embeddings = []
            all_seq_indices = []
            batch_count = 0

            for batch_info in batches:
                batch_count += 1
                features_raw = batch_info["features"]
                if "seq_indices" not in batch_info:
                    raise ValueError(
                        f"Batch {batch_count} missing required 'seq_indices' field. "
                        f"The tokenize stage must emit seq_indices for every batch so "
                        f"the embed stage can restore original chunk order."
                    )
                seq_indices = batch_info["seq_indices"]

                features = {}
                for key, padded in features_raw.items():
                    if isinstance(padded, np.ndarray):
                        features[key] = torch.from_numpy(padded).long()
                    else:
                        features[key] = torch.tensor(padded, dtype=torch.long)

                features = batch_to_device(features, self.model.device)

                with torch.no_grad():
                    out_features = self.model.forward(features)
                    embeddings = out_features["sentence_embedding"].detach()
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    embeddings = embeddings.float().cpu().numpy()
                    all_embeddings.append(embeddings)
                    all_seq_indices.append(seq_indices)
                    del out_features

                del features

                if batch_count % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if batch_count % 500 == 0:
                    logger.info(f"  Forward pass: {batch_count}/{len(batches)} batches")

            logger.info(f"Forward pass complete: {batch_count} batches processed")

            if not all_embeddings:
                return np.array([], dtype=np.float32)

            sorted_embeddings = np.concatenate(all_embeddings, axis=0)
            indices = np.concatenate(all_seq_indices, axis=0)
            result = np.empty_like(sorted_embeddings)
            result[indices] = sorted_embeddings
            logger.info(f"Unsorting embeddings: restored original order via seq_indices")
            return result

        finally:
            import shutil
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def embed_query(self, text: str) -> list:
        if self.prompt:
            text = self.prompt + text

        if not isinstance(text, str):
            text = str(text)

        text = _normalize_text(text)

        embeddings = self._safe_encode([text])
        return embeddings[0].tolist() if len(embeddings) else []

    def __del__(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None


def create_embedding_model(
    model_path: str,
    compute_device: str = "cpu",
    dtype: torch.dtype = None,
    batch_size: int = None,
    is_query: bool = False,
) -> DirectEmbeddingModel:
    config = get_config()
    model_name = os.path.basename(model_path)

    family = _get_model_family(model_path)
    model_native_precision = get_model_native_precision(model_name)

    use_half = config.database.half
    _dtype, _batch_size = get_embedding_dtype_and_batch(
        compute_device=compute_device,
        use_half=use_half,
        model_native_precision=model_native_precision,
        model_name=model_name,
        is_query=is_query,
    )

    final_dtype = dtype if dtype is not None else _dtype
    final_batch_size = batch_size if batch_size is not None else _batch_size

    if family == "qwen":
        max_seq_length = 8192
    else:
        max_seq_length = 512

    prompt = _get_prompt_for_family(family, is_query)

    return DirectEmbeddingModel(
        model_path=model_path,
        device=compute_device,
        dtype=final_dtype,
        batch_size=final_batch_size,
        max_seq_length=max_seq_length,
        prompt=prompt,
    )


def load_embedding_model(
    model_path: str,
    compute_device: str,
    use_half: bool,
    is_query: bool = False,
    verbose: bool = False,
) -> DirectEmbeddingModel:
    model_name = os.path.basename(model_path)
    model_native_precision = get_model_native_precision(model_name)

    dtype, batch_size = get_embedding_dtype_and_batch(
        compute_device=compute_device,
        use_half=use_half,
        model_native_precision=model_native_precision,
        model_name=model_name,
        is_query=is_query,
    )

    model = create_embedding_model(
        model_path=model_path,
        compute_device=compute_device,
        dtype=dtype,
        batch_size=batch_size,
        is_query=is_query,
    )

    if verbose:
        from core.utilities import my_cprint
        precision = "float32" if dtype is None else str(dtype).split(".")[-1]
        my_cprint(f"{model_name} ({precision}) loaded using a batch size of {batch_size}.", "green")

    return model
