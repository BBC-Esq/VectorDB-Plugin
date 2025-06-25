import os
import datetime
import hashlib
import re
from langchain_core.documents import Document
from typing import List, Tuple

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
    
    #=========================================================================
    clean_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            clean_metadata[k] = v
        elif isinstance(v, enumerate):
            print(f"❌ ENUMERATE in metadata key '{k}' - converting to string")
            clean_metadata[k] = str(list(v))
        else:
            clean_metadata[k] = str(v)
    
    return clean_metadata
    #=========================================================================

def extract_image_metadata(file_path):
    metadata = extract_common_metadata(file_path)
    metadata["document_type"] = "image"
    return metadata

def extract_document_metadata(file_path, content_hash=None):
    metadata = extract_common_metadata(file_path, content_hash)
    metadata["document_type"] = "document"
    return metadata

def extract_audio_metadata(file_path):
    metadata = extract_common_metadata(file_path)
    metadata["document_type"] = "audio"
    return metadata

def add_pymupdf_page_metadata(doc: Document, chunk_size: int = 1200, chunk_overlap: int = 600) -> List[Document]:
    def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int]]:
        page_markers = []
        offset = 0
        for m in re.finditer(r'\[\[page(\d+)\]\]', text):
            marker_len = len(m.group(0))
            page_markers.append((m.start() - offset, int(m.group(1))))
            offset += marker_len

        clean_text = re.sub(r'\[\[page\d+\]\]', '', text)

        chunks = []
        start = 0
        while start < len(clean_text):
            end = start + chunk_size
            if end > len(clean_text):
                end = len(clean_text)
            chunk = clean_text[start:end].strip()

            page_num = None
            for marker_pos, page in reversed(page_markers):
                if marker_pos <= start:
                    page_num = page
                    break

            if chunk and page_num is not None:
                chunks.append((chunk, page_num))
            start += chunk_size - chunk_overlap

        return chunks

    chunks = split_text(doc.page_content, chunk_size, chunk_overlap)

    #================================================================================================
    new_docs = []
    for chunk, page_num in chunks:
        new_metadata = {}
        if doc.metadata:
            for k, v in doc.metadata.items():
                if k is not None and v is not None:
                    key = str(k)
                    if isinstance(v, (str, int, float, bool)):
                        new_metadata[key] = v
                    elif isinstance(v, enumerate):
                        print(f"❌ ENUMERATE in chunk metadata key '{key}' - converting to string")
                        new_metadata[key] = str(list(v))
                    else:
                        new_metadata[key] = str(v)

        new_metadata['page_number'] = page_num

        new_doc = Document(
            page_content=str(chunk).strip(),
            metadata=new_metadata
        )
        new_docs.append(new_doc)
        #================================================================================================

    return new_docs