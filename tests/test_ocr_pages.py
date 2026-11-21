import fitz
import numpy as np
import pytest


def picture(page):
    pix = page.get_pixmap()
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(int)


def quartered_page(rotation, crop, inherited=False):
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    for index, color in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]):
        left, top = 300 * (index % 2), 400 * (index // 2)
        page.draw_rect(fitz.Rect(left, top, left + 300, top + 400), fill=color, color=None)
    if crop:
        page.set_cropbox(fitz.Rect(*crop))
    if inherited:
        parent = int(document.xref_get_key(page.xref, "Parent")[1].split()[0])
        document.xref_set_key(parent, "CropBox", document.xref_get_key(page.xref, "CropBox")[1])
        document.xref_set_key(page.xref, "CropBox", "null")
    page.set_rotation(rotation)
    return fitz.open("pdf", document.tobytes())


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("crop", [None, (0, 0, 600, 400), (300, 0, 600, 800), (100, 50, 500, 400)])
def test_removing_rotation_keeps_the_visible_part_of_the_page(rotation, crop):
    from modules.ocr import remove_page_rotation

    for inherited in ([False, True] if crop else [False]):
        source = quartered_page(rotation, crop, inherited)
        page = source[0]
        before, size = picture(page), tuple(page.rect)
        remove_page_rotation(page)
        output = fitz.open()
        output.insert_pdf(source)
        page = fitz.open("pdf", output.tobytes())[0]
        assert page.rotation == 0 and tuple(page.rect) == size
        assert np.abs(picture(page) - before).mean() < 1


def test_tesseract_output_keeps_the_title_and_author(tmp_path):
    from modules.ocr import TesseractOCR

    source, output = tmp_path / "scan.pdf", tmp_path / "scan_OCR.pdf"
    document = fitz.open()
    document.new_page()
    document.set_metadata({"title": "Lease", "author": "A. Clerk", "subject": "Renewal", "keywords": "rent"})
    document.save(source)
    pages = fitz.open()
    pages.insert_pdf(document)
    pages.save(output)
    pages.close()
    document.close()

    TesseractOCR.__new__(TesseractOCR).optimize_final_pdf(source, output)

    with fitz.open(output) as document:
        details = document.metadata
    assert [details[key] for key in ("title", "author", "subject", "keywords")] == ["Lease", "A. Clerk", "Renewal", "rent"]
    assert details["producer"].endswith("Tesseract")
