import pytest
from db.document_processor import Document

from db.document_processor import FixedSizeTextSplitter


class TestFixedSizeTextSplitter:

    def test_basic_split(self):
        doc = Document(page_content="A" * 100, metadata={'file': 'test.txt'})
        splitter = FixedSizeTextSplitter(chunk_size=30, chunk_overlap=0)
        chunks = splitter.split_documents([doc])
        assert len(chunks) == 4

    def test_no_overlap_no_repeated_text(self):
        text = "ABCDEFGHIJ" * 10
        doc = Document(page_content=text, metadata={})
        splitter = FixedSizeTextSplitter(chunk_size=25, chunk_overlap=0)
        chunks = splitter.split_documents([doc])
        combined = "".join(c.page_content for c in chunks)
        assert combined == text

    def test_with_overlap(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        doc = Document(page_content=text, metadata={})
        splitter = FixedSizeTextSplitter(chunk_size=10, chunk_overlap=3)
        chunks = splitter.split_documents([doc])
        assert len(chunks) >= 3
        assert chunks[0].page_content[-3:] == chunks[1].page_content[:3]

    def test_single_chunk(self):
        doc = Document(page_content="Short text", metadata={})
        splitter = FixedSizeTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text"

    def test_empty_document(self):
        doc = Document(page_content="", metadata={})
        splitter = FixedSizeTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_documents([doc])
        assert chunks == []

    def test_metadata_preserved(self):
        doc = Document(page_content="A" * 50, metadata={'file_name': 'test.txt', 'hash': 'abc'})
        splitter = FixedSizeTextSplitter(chunk_size=20, chunk_overlap=0)
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata['file_name'] == 'test.txt'
            assert chunk.metadata['hash'] == 'abc'

    def test_metadata_independence(self):
        doc = Document(page_content="A" * 50, metadata={'key': 'value'})
        splitter = FixedSizeTextSplitter(chunk_size=20, chunk_overlap=0)
        chunks = splitter.split_documents([doc])
        chunks[0].metadata['key'] = 'changed'
        assert chunks[1].metadata['key'] == 'value'

    def test_multiple_documents(self):
        docs = [
            Document(page_content="A" * 30, metadata={'file': 'a'}),
            Document(page_content="B" * 30, metadata={'file': 'b'}),
            Document(page_content="C" * 30, metadata={'file': 'c'}),
        ]
        splitter = FixedSizeTextSplitter(chunk_size=20, chunk_overlap=0)
        chunks = splitter.split_documents(docs)
        assert len(chunks) == 6

    def test_exact_chunk_size(self):
        doc = Document(page_content="A" * 50, metadata={})
        splitter = FixedSizeTextSplitter(chunk_size=50, chunk_overlap=0)
        chunks = splitter.split_documents([doc])
        assert len(chunks) == 1
