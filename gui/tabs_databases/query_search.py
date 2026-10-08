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


def chunks_query(database_name, query, result_queue):
    try:
        from core.utilities import configure_logging

        configure_logging("INFO")
        from db.database_interactions import QueryVectorDB

        query_db = QueryVectorDB(database_name)
        try:
            contexts, metadata_list = query_db.search(query)
        finally:
            query_db.close()
        result_queue.put({"chunks": chunk_rows(contexts, metadata_list)})
    except Exception as e:
        result_queue.put({"error": str(e)})
