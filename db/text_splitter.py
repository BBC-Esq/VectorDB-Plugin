"""Pure-Python text-splitting primitives (stdlib + re only).

Split out of db.document_processor so the headless split worker
(db/stage_split.py) can import these without dragging in fitz + bs4, which
document_processor imports at module top for its document loaders.
db.document_processor re-exports these names for backwards compatibility.

Do not add fitz / bs4 / torch / Qt imports here.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Document:
    page_content: str = ""
    metadata: dict = field(default_factory=dict)


class FixedSizeTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, docs: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = 1

        for doc in docs:
            text = doc.page_content

            if text is None:
                logger.warning("Skipping document with None page_content")
                continue

            if isinstance(text, (list, tuple)):
                text = " ".join(str(item) for item in text if item)
                logger.warning("Flattened list/tuple page_content to string")

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if not text:
                logger.warning("Skipping document with empty page_content")
                continue

            for start in range(0, len(text), step):
                piece = text[start:start + self.chunk_size].strip()

                if not piece:
                    continue

                metadata = doc.metadata if doc.metadata else {}
                chunks.append(Document(page_content=piece, metadata=dict(metadata)))

        return chunks


def add_pymupdf_page_metadata(doc: Document, chunk_size: int = 1200, chunk_overlap: int = 600) -> List[Document]:
    def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int]]:
        if text is None:
            return []

        if isinstance(text, (list, tuple)):
            text = " ".join(str(item) for item in text if item)

        if not isinstance(text, str):
            text = str(text)

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
            elif chunk and page_num is None:
                chunks.append((chunk, 1))

            start += chunk_size - chunk_overlap

        return chunks

    text = doc.page_content

    if text is None:
        logger.warning("Skipping PDF document with None page_content")
        return []

    chunks = split_text(text, chunk_size, chunk_overlap)

    if not chunks:
        logger.warning("No chunks created from PDF document")
        return []

    new_docs = []
    for chunk, page_num in chunks:
        if not chunk or not chunk.strip():
            continue

        new_metadata = doc.metadata.copy() if doc.metadata else {}
        new_metadata['page_number'] = page_num

        new_doc = Document(page_content=chunk, metadata=new_metadata)
        new_docs.append(new_doc)

    return new_docs
