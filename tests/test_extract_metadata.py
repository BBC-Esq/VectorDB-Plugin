import hashlib
import os

import pytest
from db.document_processor import Document

from core.extract_metadata import (
    compute_content_hash,
    compute_file_hash,
    extract_common_metadata,
    extract_typed_metadata,
    add_pymupdf_page_metadata,
)


class TestComputeContentHash:

    def test_deterministic(self):
        h1 = compute_content_hash("hello world")
        h2 = compute_content_hash("hello world")
        assert h1 == h2

    def test_different_inputs(self):
        h1 = compute_content_hash("hello")
        h2 = compute_content_hash("world")
        assert h1 != h2

    def test_matches_hashlib(self):
        text = "test content"
        expected = hashlib.sha256(text.encode('utf-8')).hexdigest()
        assert compute_content_hash(text) == expected

    def test_empty_string(self):
        result = compute_content_hash("")
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_unicode(self):
        result = compute_content_hash("caf\u00e9 \u2603")
        assert isinstance(result, str)
        assert len(result) == 64


class TestComputeFileHash:

    def test_matches_content(self, tmp_path):
        content = b"some file content"
        f = tmp_path / "test.txt"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_file_hash(str(f)) == expected

    def test_binary_file(self, tmp_path):
        content = bytes(range(256))
        f = tmp_path / "binary.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_file_hash(str(f)) == expected

    def test_large_file(self, tmp_path):
        content = b"x" * 10000
        f = tmp_path / "large.txt"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert compute_file_hash(str(f)) == expected


class TestExtractCommonMetadata:

    def test_keys(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello", encoding='utf-8')
        meta = extract_common_metadata(str(f))
        expected_keys = {'file_path', 'file_type', 'file_name', 'creation_date', 'modification_date', 'hash'}
        assert expected_keys == set(meta.keys())

    def test_file_type(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello", encoding='utf-8')
        meta = extract_common_metadata(str(f))
        assert meta['file_type'] == '.txt'

    def test_file_name(self, tmp_path):
        f = tmp_path / "my_document.pdf"
        f.write_bytes(b"%PDF-1.4")
        meta = extract_common_metadata(str(f))
        assert meta['file_name'] == 'my_document.pdf'

    def test_with_content_hash(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello", encoding='utf-8')
        custom_hash = "abc123"
        meta = extract_common_metadata(str(f), content_hash=custom_hash)
        assert meta['hash'] == custom_hash

    def test_values_serializable(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello", encoding='utf-8')
        meta = extract_common_metadata(str(f))
        for v in meta.values():
            assert isinstance(v, (str, int, float, bool, type(None)))


class TestExtractTypedMetadata:

    def test_adds_document_type(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello", encoding='utf-8')
        meta = extract_typed_metadata(str(f), "document")
        assert meta['document_type'] == 'document'

    def test_preserves_common_keys(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello", encoding='utf-8')
        meta = extract_typed_metadata(str(f), "image")
        assert 'file_path' in meta
        assert 'hash' in meta


class TestAddPymupdfPageMetadata:

    def test_basic_chunking(self):
        doc = Document(page_content="[[page1]]Hello world this is a test document.", metadata={'file_name': 'test.pdf'})
        chunks = add_pymupdf_page_metadata(doc, chunk_size=50, chunk_overlap=0)
        assert len(chunks) >= 1
        assert chunks[0].metadata['page_number'] == 1

    def test_multi_page_markers(self):
        content = "[[page1]]First page content. " + "[[page2]]Second page content."
        doc = Document(page_content=content, metadata={})
        chunks = add_pymupdf_page_metadata(doc, chunk_size=20, chunk_overlap=0)
        pages = [c.metadata['page_number'] for c in chunks]
        assert 1 in pages
        assert 2 in pages

    def test_chunk_size_respected(self):
        content = "[[page1]]" + "A" * 200
        doc = Document(page_content=content, metadata={})
        chunks = add_pymupdf_page_metadata(doc, chunk_size=50, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.page_content) <= 50

    def test_chunk_overlap(self):
        content = "[[page1]]" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 5
        doc = Document(page_content=content, metadata={})
        chunks = add_pymupdf_page_metadata(doc, chunk_size=30, chunk_overlap=10)
        if len(chunks) >= 2:
            overlap_text = chunks[0].page_content[-10:]
            assert overlap_text in chunks[1].page_content

    def test_metadata_preserved(self):
        doc = Document(page_content="[[page1]]Some text here", metadata={'file_name': 'test.pdf', 'hash': 'abc123'})
        chunks = add_pymupdf_page_metadata(doc, chunk_size=100, chunk_overlap=0)
        assert chunks[0].metadata['file_name'] == 'test.pdf'
        assert chunks[0].metadata['hash'] == 'abc123'
        assert 'page_number' in chunks[0].metadata

    def test_empty_content(self):
        doc = Document(page_content="", metadata={})
        chunks = add_pymupdf_page_metadata(doc, chunk_size=100, chunk_overlap=0)
        assert chunks == []

    def test_markers_stripped_from_content(self):
        doc = Document(page_content="[[page1]]Hello world", metadata={})
        chunks = add_pymupdf_page_metadata(doc, chunk_size=100, chunk_overlap=0)
        assert '[[page' not in chunks[0].page_content
