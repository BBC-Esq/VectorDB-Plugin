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

    def test_overlap_adds_no_chunk_that_only_repeats_the_end(self):
        for length in (451, 600, 700, 701, 901, 1150, 1151, 1600, 2500):
            text = "".join(chr(97 + i % 26) for i in range(length))
            chunks = FixedSizeTextSplitter(chunk_size=700, chunk_overlap=250).split_documents(
                [Document(page_content=text, metadata={})])
            pieces = [c.page_content for c in chunks]
            assert "".join([pieces[0]] + [piece[250:] for piece in pieces[1:]]) == text
            assert len(pieces) == max(1, -(-(length - 700) // 450) + 1)

    def test_pdf_overlap_adds_no_chunk_that_only_repeats_the_end(self):
        from db.text_splitter import add_pymupdf_page_metadata

        for length in (451, 600, 901, 1150, 1600):
            text = "[[page1]]" + "".join(chr(97 + i % 26) for i in range(length))
            pieces = [c.page_content for c in add_pymupdf_page_metadata(
                Document(page_content=text, metadata={}), chunk_size=700, chunk_overlap=250)]
            assert "".join([pieces[0]] + [piece[250:] for piece in pieces[1:]]) == text[9:]
            assert len(pieces) == max(1, -(-(length - 700) // 450) + 1)

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


class TestEmailLoader:

    def _load(self, tmp_path, raw):
        from db.document_processor import _load_eml

        path = tmp_path / "message.eml"
        path.write_bytes(raw)
        return _load_eml(path)

    def test_html_body_is_read_when_the_plain_part_is_empty(self, tmp_path):
        raw = (b"Subject: Minutes\r\nMIME-Version: 1.0\r\n"
               b'Content-Type: multipart/alternative; boundary="b"\r\n\r\n'
               b"--b\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n \r\n"
               b"--b\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
               b"<html><body><p>The committee voted to repaint the lantern room.</p></body></html>\r\n--b--\r\n")
        text = self._load(tmp_path, raw)
        assert text.startswith("Subject: Minutes") and "voted to repaint the lantern room" in text

    def test_plain_part_is_preferred_when_it_has_text(self, tmp_path):
        raw = (b"Subject: Minutes\r\nMIME-Version: 1.0\r\n"
               b'Content-Type: multipart/alternative; boundary="b"\r\n\r\n'
               b"--b\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nPlain version.\r\n"
               b"--b\r\nContent-Type: text/html; charset=utf-8\r\n\r\n<p>HTML version.</p>\r\n--b--\r\n")
        text = self._load(tmp_path, raw)
        assert "Plain version." in text and "HTML version." not in text

    @pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
    def test_accented_letters_survive_a_missing_charset(self, tmp_path, encoding):
        body = "Señor Núñez ordered crème brûlée at the café."
        raw = b"Subject: Menu\r\nContent-Type: text/plain\r\nContent-Transfer-Encoding: 8bit\r\n\r\n" + body.encode(encoding)
        assert body in self._load(tmp_path, raw)
