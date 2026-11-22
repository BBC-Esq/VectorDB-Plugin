import json
import os
from types import SimpleNamespace

import fitz
import pytest

import core.ocr_cache as ocr_cache
import db.document_processor as document_processor
import db.stage_ocr as stage_ocr
from gui.tabs_databases import create_build, create_files
from modules import ocr_reader

LINES = ["The first typed line of the page.", "The second typed line of the page.",
         "The third typed line of the page.", "The fourth typed line of the page."] * 3


def typed_page(document):
    page = document.new_page(width=612, height=792)
    for i, line in enumerate(LINES):
        page.insert_text((100, 150 + 16 * i), line, fontsize=11)
    return page


def scanned_page(document):
    page = document.new_page(width=612, height=792)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 306, 396), False)
    pixmap.clear_with(235)
    page.insert_image(page.rect, pixmap=pixmap)
    return page


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_cache, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(stage_ocr, "worker_count", lambda tasks: 1)
    docs = tmp_path / "docs"
    docs.mkdir()
    with fitz.open() as document:
        typed_page(document)
        document.save(docs / "typed.pdf")
    with fitz.open() as document:
        typed_page(document)
        scanned_page(document)
        typed_page(document)
        document.save(docs / "mixed.pdf")
    with fitz.open() as document:
        typed_page(document)
        document.save(docs / "locked.pdf", encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    (docs / "broken.pdf").write_bytes(b"not a pdf")
    return docs


def test_the_scan_lists_pages_that_need_ocr_and_pdfs_that_cannot_be_read(folder):
    found = stage_ocr.scan(folder, "quick", "rapidocr")
    assert [(item["name"], item["pages"], item["total"], item["unread"]) for item in found["needs"]] == [("mixed.pdf", 1, 3, 1)]
    assert {item["name"]: item["reason"] for item in found["problems"]} == {
        "broken.pdf": "it could not be opened", "locked.pdf": "it is password protected"}


def test_a_second_scan_reuses_the_saved_page_check(folder, monkeypatch):
    first = stage_ocr.scan(folder, "quick", "rapidocr")
    monkeypatch.setattr(stage_ocr, "run_tasks", lambda *args: pytest.fail("the pages were checked again"))
    assert stage_ocr.scan(folder, "quick", "rapidocr") == first


def test_a_changed_pdf_is_checked_again(folder):
    stage_ocr.scan(folder, "quick", "rapidocr")
    with fitz.open() as document:
        scanned_page(document)
        scanned_page(document)
        document.save(folder / "typed.pdf")
    found = stage_ocr.scan(folder, "quick", "rapidocr")
    assert {item["name"]: item["pages"] for item in found["needs"]} == {"mixed.pdf": 1, "typed.pdf": 2}


def test_reading_saves_the_text_and_does_not_read_a_page_twice(folder, monkeypatch):
    calls = []

    def fake_tasks(function, tasks, option, check, reader, label):
        if function is ocr_reader.read_pages:
            calls.append(tasks)
            return [(path, pages, {page: f"ocr text of page {page}" for page in pages}) for path, pages in tasks]
        return [function(path, pages, option) for path, pages in tasks]

    monkeypatch.setattr(stage_ocr, "run_tasks", fake_tasks)
    monkeypatch.setattr(ocr_reader, "ensure_models", lambda check, reader: None)
    expected = {os.path.normcase(os.path.realpath(folder / "mixed.pdf")): {"2": "ocr text of page 2"}}
    assert stage_ocr.read(folder, "quick", "rapidocr") == expected
    assert stage_ocr.read(folder, "quick", "rapidocr") == expected
    assert len(calls) == 1
    assert stage_ocr.scan(folder, "quick", "rapidocr")["needs"][0]["unread"] == 0


def test_the_pdf_loader_uses_ocr_text_under_the_original_page_numbers(folder, tmp_path, monkeypatch):
    handoff = tmp_path / "ocr_text.json"
    handoff.write_text(json.dumps({os.path.normcase(os.path.realpath(folder / "mixed.pdf")): {"2": "text read by OCR"}}),
                       encoding="utf-8")
    monkeypatch.setenv(document_processor.OCR_TEXT_VARIABLE, str(handoff))
    monkeypatch.setattr(document_processor, "_ocr_text", None)
    content = document_processor._load_pdf(folder / "mixed.pdf")
    assert content.count("[[page") == 3
    assert "[[page2]]text read by OCR\n[[page3]]The first typed line" in content
    assert "text read by OCR" not in document_processor._load_pdf(folder / "typed.pdf")


def test_the_pdf_loader_reads_pdfs_as_before_without_ocr_text(folder, monkeypatch):
    monkeypatch.delenv(document_processor.OCR_TEXT_VARIABLE, raising=False)
    monkeypatch.setattr(document_processor, "_ocr_text", None)
    content = document_processor._load_pdf(folder / "mixed.pdf")
    assert content.count("[[page") == 2 and "[[page2]]" not in content


class Detector:
    def __init__(self, boxes):
        self.boxes = boxes

    def __call__(self, image, **flags):
        assert flags == ocr_reader.DETECT_ONLY
        return SimpleNamespace(boxes=self.boxes)


def box(x0, y0, x1, y1, dpi=ocr_reader.GATE_DPI):
    scale = dpi / 72.0
    return [[x0 * scale, y0 * scale], [x1 * scale, y0 * scale], [x1 * scale, y1 * scale], [x0 * scale, y1 * scale]]


def test_the_detector_check_keeps_a_page_whose_lines_are_all_its_own_text():
    with fitz.open() as document:
        page = typed_page(document)
        lines = [box(95, 138 + 16 * i, 330, 154 + 16 * i) for i in range(len(LINES))]
        assert not ocr_reader.page_requires_ocr(page, "detector", Detector(lines))


def test_the_detector_check_flags_a_line_the_pdf_holds_no_text_for():
    with fitz.open() as document:
        page = typed_page(document)
        lines = [box(95, 138, 330, 154), box(350, 600, 560, 620)]
        assert ocr_reader.page_requires_ocr(page, "detector", Detector(lines))


def test_the_detector_check_does_not_count_hidden_text():
    with fitz.open() as document:
        page = scanned_page(document)
        page.insert_text((100, 150), "hidden text from an earlier OCR", fontsize=11, render_mode=3)
        assert ocr_reader.page_requires_ocr(page, "detector", Detector([box(95, 138, 330, 154)]))
        assert not ocr_reader.page_requires_ocr(page, "detector", Detector([box(95, 138, 330, 154)]), own=True)


def test_the_detector_check_leaves_a_blank_page_alone():
    with fitz.open() as document:
        assert not ocr_reader.page_requires_ocr(document.new_page(), "detector", Detector(None))


@pytest.fixture
def controller(tmp_path, monkeypatch):
    for module in (create_build, create_files):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "Docs_for_DB").mkdir()
    (tmp_path / "Docs_for_DB" / "scan.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "Docs_for_DB" / "locked.pdf").write_text("x", encoding="utf-8")
    controller = create_build.BuildController(None, create_files.StagedFiles(None))
    started = []

    class Worker:
        progress = finished = SimpleNamespace(connect=lambda slot: None)

        def start(self):
            started.append(controller.ocr)

    monkeypatch.setattr(controller, "make_worker", lambda name: Worker())
    monkeypatch.setattr(create_build, "current_build_settings", lambda: {})
    monkeypatch.setattr(create_build, "my_cprint", lambda *args: None)
    controller.name = "db"
    controller.phase = "scanning"
    controller.started_builds = started
    return controller


FOUND = {"needs": [{"name": "scan.pdf", "path": "C:/docs/scan.pdf", "pages": 3, "total": 4, "unread": 2}],
         "problems": [{"name": "locked.pdf", "path": "C:/docs/locked.pdf", "reason": "it is password protected"}]}


def test_pages_that_need_ocr_wait_for_approval_before_the_build(controller):
    controller._on_scanned(FOUND)
    assert controller.phase == "idle" and controller.started_builds == []
    result = controller.result
    assert (result["kind"], result["pages"], result["unread"], result["reader"]) == ("ocr", 3, 2, "RapidOCR")
    assert result["names"] == ["scan.pdf", "locked.pdf"]
    assert controller.approve_ocr() == {"ok": True}
    assert controller.phase == "building" and controller.started_builds == [True]
    assert [stage["key"] for stage in controller.progress.state()["stages"]][0] == "ocr"


def test_a_build_with_only_unreadable_pdfs_runs_without_ocr(controller):
    controller._on_scanned({"needs": [], "problems": FOUND["problems"]})
    assert controller.result["kind"] == "ocr" and controller.started_builds == []
    controller.approve_ocr()
    assert controller.started_builds == [False]
    assert "ocr" not in [stage["key"] for stage in controller.progress.state()["stages"]]


def test_a_clean_scan_starts_the_build_without_asking(controller):
    controller._on_scanned({"needs": [], "problems": []})
    assert controller.phase == "building" and controller.started_builds == [False]


def test_a_failed_scan_creates_no_database(controller):
    controller._on_scanned(None)
    assert controller.phase == "idle" and controller.result["kind"] == "error" and controller.started_builds == []


def test_declining_removes_the_listed_pdfs(controller):
    controller._on_scanned(FOUND)
    controller.remove_ocr_files()
    assert controller.files.names == [] and controller.result["kind"] == "info"
    assert controller.approve_ocr() == {"error": "There is nothing to continue."}


def test_build_progress_follows_the_ocr_stage():
    progress = create_build.BuildProgress(ocr=True)
    progress.feed("Reading scanned pages with OCR (subprocess)...")
    progress.feed("2026-11-22 INFO [db]   [OCR] Reading pages with OCR: 16/40 pages")
    state = progress.state()
    assert state["stage"] == "ocr" and state["ocr_pages"] == [16, 40]
    progress.feed("Extracting documents (subprocess)...")
    assert progress.state()["stages"][0] == {"key": "ocr", "label": "OCR", "status": "done"}


def test_ocr_settings_fall_back_to_the_defaults():
    assert create_build.ocr_settings({}) == ("detector", "rapidocr")
    assert create_build.ocr_settings({"database": {"ocr_check": "quick", "ocr_reader": "tesseract"}}) == ("quick", "tesseract")
    assert create_build.ocr_settings({"database": {"ocr_check": "x", "ocr_reader": None}}) == ("detector", "rapidocr")
