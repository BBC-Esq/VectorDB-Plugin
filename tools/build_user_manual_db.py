from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ASSETS = PROJECT_ROOT / "Assets"
CONSOLIDATED_MD = ASSETS / "user_manual_consolidated.md"
CHUNKS_DIR = ASSETS / "User_Guide_Chunks"
OUTPUT_ZIP = ASSETS / "user_manual_db.zip"
VECTOR_DB = PROJECT_ROOT / "Vector_DB"
VECTOR_DB_BACKUP = PROJECT_ROOT / "Vector_DB_Backup"
DB_NAME = "user_manual"
BGE_MODEL = PROJECT_ROOT / "Models" / "vector" / "BAAI--bge-small-en-v1.5"
BGE_MAX_TOKENS = 512


def extract_chunks(content: str) -> list[str]:
    chunks: list[str] = []
    current_chunk: list[str] = []
    for line in content.split("\n"):
        if line.strip().startswith("###"):
            if current_chunk:
                text = "\n".join(current_chunk).strip()
                if text:
                    chunks.append(text)
            current_chunk = [line]
        elif current_chunk:
            if line.strip() or len(current_chunk) == 1:
                current_chunk.append(line)
            else:
                continue
    if current_chunk:
        text = "\n".join(current_chunk).strip()
        if text:
            chunks.append(text)
    return chunks


def write_chunk_files(chunks: list[str]) -> list[Path]:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for existing in CHUNKS_DIR.iterdir():
        if existing.is_file():
            existing.unlink()
    paths: list[Path] = []
    for i, chunk in enumerate(chunks, 1):
        path = CHUNKS_DIR / f"chunk_{i:03d}.txt"
        path.write_text(chunk, encoding="utf-8")
        paths.append(path)
    return paths


def find_oversize_chunks(chunks: list[str], chunk_paths: list[Path]):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(BGE_MODEL))
        measure = lambda text: len(tokenizer.encode(text, add_special_tokens=True))
        unit, limit = "tokens", BGE_MAX_TOKENS
    except Exception:
        measure = len
        unit, limit = "chars (approx)", BGE_MAX_TOKENS * 4
    oversize = [(p.name, measure(c)) for c, p in zip(chunks, chunk_paths) if measure(c) > limit]
    return oversize, unit, limit


def build_database(chunk_paths: list[Path], persist_dir: Path) -> int:
    import gc

    import numpy as np
    import torch

    from core.utilities import my_cprint, set_cuda_paths
    set_cuda_paths()

    from db.database_interactions import CreateVectorDB
    from db.document_processor import Document, extract_document_metadata
    from db.embedding_models import load_embedding_model
    from db.sqlite_operations import create_metadata_db

    texts = [p.read_text(encoding="utf-8") for p in chunk_paths]
    metadatas = [extract_document_metadata(str(p)) for p in chunk_paths]

    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    my_cprint(f"Loading BGE-small on {compute_device} ...", "yellow")
    embeddings = load_embedding_model(
        model_path=str(BGE_MODEL),
        compute_device=compute_device,
        use_half=False,
        is_query=False,
        verbose=True,
    )
    try:
        with torch.inference_mode():
            vectors, surviving = embeddings.embed_documents(texts)
    finally:
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if len(surviving) != len(texts):
        my_cprint(f"Warning: {len(texts) - len(surviving)} chunk(s) failed tokenization; skipped.", "red")
        texts = [texts[i] for i in surviving]
        metadatas = [metadatas[i] for i in surviving]

    for doc_id, metadata in enumerate(metadatas, start=1):
        metadata["doc_id"] = doc_id

    vectors_array = np.ascontiguousarray(vectors, dtype=np.float32)

    persist_dir.mkdir(parents=True, exist_ok=True)
    builder = CreateVectorDB.__new__(CreateVectorDB)
    builder.PERSIST_DIRECTORY = persist_dir
    hash_id_mappings = builder._create_tiledb_array(texts, vectors_array, metadatas)

    documents = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    create_metadata_db(persist_dir, documents, hash_id_mappings)
    return len(texts)


def zip_database(persist_dir: Path, output_zip: Path) -> None:
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(persist_dir.rglob("*")):
            if path.is_file():
                arcname = (Path(DB_NAME) / path.relative_to(persist_dir)).as_posix()
                zf.write(path, arcname)


def deploy_database(persist_dir: Path) -> None:
    for target_root in (VECTOR_DB, VECTOR_DB_BACKUP):
        target_root.mkdir(parents=True, exist_ok=True)
        dest = target_root / DB_NAME
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(persist_dir, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Ask Jeeves user_manual vector database.")
    parser.add_argument("--no-deploy", action="store_true",
                        help="Build and zip only; do not copy into Vector_DB / Vector_DB_Backup.")
    args = parser.parse_args()

    if not CONSOLIDATED_MD.exists():
        sys.exit(f"ERROR: {CONSOLIDATED_MD} not found.")
    if not BGE_MODEL.is_dir():
        sys.exit(f"ERROR: BGE-small model not found at {BGE_MODEL}.")

    chunks = extract_chunks(CONSOLIDATED_MD.read_text(encoding="utf-8"))
    if not chunks:
        sys.exit("ERROR: No '###' chunks found in the consolidated manual.")

    chunk_paths = write_chunk_files(chunks)
    lengths = [len(c) for c in chunks]
    print(f"Split into {len(chunks)} chunk(s) -> {CHUNKS_DIR}")
    print(f"  longest: {max(lengths)} chars | shortest: {min(lengths)} chars")

    oversize, unit, limit = find_oversize_chunks(chunks, chunk_paths)
    if oversize:
        print(f"  WARNING: {len(oversize)} chunk(s) exceed {limit} {unit} (BGE-small will truncate them):")
        for name, n in oversize:
            print(f"    {name}: {n} {unit}")

    staging = Path(tempfile.mkdtemp(prefix="um_build_"))
    persist_dir = staging / DB_NAME
    try:
        count = build_database(chunk_paths, persist_dir)
        print(f"Built new-format database: {count} vector(s)")

        zip_database(persist_dir, OUTPUT_ZIP)
        print(f"Wrote {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 1024:.0f} KB)")

        if args.no_deploy:
            print("Skipped deploy (--no-deploy). Run setup, or re-run without the flag, to install it.")
        else:
            try:
                deploy_database(persist_dir)
                print(f"Deployed to {VECTOR_DB / DB_NAME} and {VECTOR_DB_BACKUP / DB_NAME}")
            except Exception as exc:
                print(f"WARNING: deploy failed ({exc}). The zip is written; close the app and re-run.")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
