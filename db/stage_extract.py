import logging
import os
import pickle
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("stage_extract")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <output_pickle>", file=sys.stderr)
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not source_dir.is_dir():
        print(f"ERROR: Source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Stage 1: Extracting documents from {source_dir}")
    t0 = time.time()

    from db.document_processor import load_documents

    docs = load_documents(source_dir)

    doc_data = []
    for doc in docs:
        clean_meta = {str(k): v for k, v in doc.metadata.items()}
        doc_data.append((doc.page_content, clean_meta))

    elapsed = time.time() - t0
    logger.info(f"Extracted {len(doc_data)} documents in {elapsed:.1f}s")

    with open(output_path, "wb") as f:
        pickle.dump(doc_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Wrote {output_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
