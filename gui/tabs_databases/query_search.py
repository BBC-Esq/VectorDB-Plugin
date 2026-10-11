import re
from pathlib import Path


def clean_chunk(text):
    text = re.sub(r"\n[ \t]+\n", "\n\n", text or "")
    return re.sub(r"\n\s*\n\s*\n*", "\n\n", text.strip())


def chunk_rows(contexts, metadata_list):
    rows = []
    for text, meta in zip(contexts, metadata_list):
        path = str(meta.get("file_path") or "")
        rows.append({
            "text": clean_chunk(text),
            "name": str(meta.get("file_name") or "") or Path(path).name or "Unknown file",
            "path": path,
            "score": meta.get("similarity_score"),
            "page": meta.get("page_number"),
        })
    return rows
