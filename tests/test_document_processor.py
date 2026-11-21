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


class TestTextLoaders:

    def test_one_stray_byte_does_not_garble_a_utf8_file(self):
        from db.document_processor import _decode_text

        text = "The café’s résumé policy — naïve façade. " * 5
        damaged = text.encode("utf-8") + b"It\x92s final."
        assert _decode_text(damaged) == text + "It’s final."
        assert _decode_text(text.encode("utf-8")) == text

    def test_a_windows_1252_file_is_still_read_as_windows_1252(self):
        from db.document_processor import _decode_text

        text = "Señor Núñez ordered crème brûlée at the café."
        assert _decode_text(text.encode("cp1252")) == text

    def test_csv_with_a_very_long_cell_loads(self, tmp_path):
        from db.document_processor import _load_csv

        (tmp_path / "long.csv").write_text("id,notes\n1," + "word " * 40000 + "\n", encoding="utf-8")
        text = _load_csv(tmp_path / "long.csv")
        assert text.startswith("id notes") and len(text) > 200000

    def test_a_word_text_box_is_read_once(self, tmp_path):
        import zipfile

        from db.document_processor import _load_docx

        box = "<w:txbxContent><w:p><w:r><w:t>BOX TEXT</w:t></w:r></w:p></w:txbxContent>"
        document = (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
            'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
            'xmlns:v="urn:schemas-microsoft-com:vml"><w:body><w:p><w:r><w:t>Before.</w:t></w:r></w:p><w:p><w:r>'
            f'<mc:AlternateContent><mc:Choice Requires="wps"><wps:txbx>{box}</wps:txbx></mc:Choice>'
            f"<mc:Fallback><v:textbox>{box}</v:textbox></mc:Fallback></mc:AlternateContent>"
            "</w:r></w:p></w:body></w:document>"
        )
        with zipfile.ZipFile(tmp_path / "box.docx", "w") as archive:
            archive.writestr("word/document.xml", document)
        text = _load_docx(tmp_path / "box.docx")
        assert "Before." in text and text.count("BOX TEXT") == 1


class TestSpreadsheetLoaders:

    ROWS = [["Supplier", "Item"], ["Halvorsen Marine", "bilge pump"], ["Okonkwo Textiles", "indigo dye"]]
    TEXT = "Supplier Item\nHalvorsen Marine bilge pump\nOkonkwo Textiles indigo dye"

    def _workbook(self, path, chart_sheet=False):
        from openpyxl import Workbook

        workbook = Workbook()
        for row in self.ROWS:
            workbook.active.append(row)
        if chart_sheet:
            from openpyxl.chart import BarChart, Reference

            chart = BarChart()
            chart.add_data(Reference(workbook.active, min_col=2, min_row=1, max_row=3))
            workbook.create_chartsheet("Chart").add_chart(chart)
        workbook.save(path)

    def test_xlsx_with_a_chart_sheet_still_loads_its_data(self, tmp_path):
        from db.document_processor import _load_xlsx

        self._workbook(tmp_path / "chart.xlsx", chart_sheet=True)
        assert _load_xlsx(tmp_path / "chart.xlsx") == self.TEXT

    def test_xlsx_with_a_wrong_size_tag_loads_every_row(self, tmp_path):
        import zipfile

        from db.document_processor import _load_xlsx

        self._workbook(tmp_path / "source.xlsx")
        with zipfile.ZipFile(tmp_path / "source.xlsx") as source, \
                zipfile.ZipFile(tmp_path / "small_tag.xlsx", "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "xl/worksheets/sheet1.xml":
                    assert b'<dimension ref="A1:B3"' in data
                    data = data.replace(b'<dimension ref="A1:B3"', b'<dimension ref="A1"')
                target.writestr(item, data)
        assert _load_xlsx(tmp_path / "small_tag.xlsx") == self.TEXT

    def test_xls_numbers_and_dates_are_written_like_xlsx(self):
        import xlrd
        from types import SimpleNamespace

        from db.document_processor import _xls_cell_text

        def cell(ctype, value):
            return _xls_cell_text(SimpleNamespace(ctype=ctype, value=value), 0)

        assert cell(xlrd.XL_CELL_NUMBER, 1042.0) == "1042"
        assert cell(xlrd.XL_CELL_NUMBER, 2.5) == "2.5"
        assert cell(xlrd.XL_CELL_DATE, 45352.0) == "2024-03-01 00:00:00"
        assert cell(xlrd.XL_CELL_TEXT, "Invoice") == "Invoice"
