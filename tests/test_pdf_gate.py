import fitz

from core.pdf_ocr_gate import document_needs_ocr, page_needs_ocr


def _scan_page(doc):
    page = doc.new_page(width=612, height=792)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 306, 396), False)
    pixmap.clear_with(235)
    page.insert_image(page.rect, pixmap=pixmap)
    return page


def _typed_lines(page, count, hidden=False):
    for i in range(count):
        page.insert_text((100, 150 + 14 * i), f"line {i} of the typed text on this page", fontsize=10,
                         render_mode=3 if hidden else 0)


def test_a_scan_with_a_few_stamped_lines_still_needs_ocr():
    with fitz.open() as doc:
        page = _scan_page(doc)
        page.insert_text((200, 300), "FILED IN OFFICE 3/4/2026 CLERK OF COURT", fontsize=10)
        page.insert_text((200, 314), "E-FILED CASE NUMBER 26-A-01234", fontsize=10)
        assert page_needs_ocr(page)


def test_a_scan_with_no_text_needs_ocr():
    with fitz.open() as doc:
        assert page_needs_ocr(_scan_page(doc))


def test_a_typed_page_does_not_need_ocr():
    with fitz.open() as doc:
        page = doc.new_page(width=612, height=792)
        _typed_lines(page, 30)
        assert not page_needs_ocr(page)


def test_a_full_page_image_under_many_typed_lines_does_not_need_ocr():
    with fitz.open() as doc:
        page = _scan_page(doc)
        _typed_lines(page, 30)
        assert not page_needs_ocr(page)


def test_a_scan_that_already_has_a_hidden_text_layer_does_not_need_ocr():
    with fitz.open() as doc:
        page = _scan_page(doc)
        _typed_lines(page, 2, hidden=True)
        assert not page_needs_ocr(page)
        assert not document_needs_ocr(doc)


def test_a_blank_page_does_not_need_ocr():
    with fitz.open() as doc:
        assert not page_needs_ocr(doc.new_page())
