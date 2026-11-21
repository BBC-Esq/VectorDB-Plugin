import os
import datetime
import hashlib

def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def compute_file_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def extract_common_metadata(file_path, content_hash=None):
    file_path = os.path.realpath(file_path)
    file_name = os.path.basename(file_path)
    file_type = os.path.splitext(file_path)[1]
    creation_date = datetime.datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    modification_date = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    file_hash = content_hash if content_hash else compute_file_hash(file_path)

    metadata = {
        "file_path": file_path,
        "file_type": file_type,
        "file_name": file_name,
        "creation_date": creation_date,
        "modification_date": modification_date,
        "hash": file_hash
    }

    clean_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            clean_metadata[k] = v
        else:
            clean_metadata[k] = str(v)

    return clean_metadata

def extract_typed_metadata(file_path, document_type, content_hash=None):
    metadata = extract_common_metadata(file_path, content_hash)
    metadata["document_type"] = document_type
    return metadata
