import queue as queue_module
import time
import uuid
from datetime import date

import pytest
import yaml
from PySide6.QtCore import QSettings

import core.utilities as utilities
from gui.tabs_tools import backups, gpus, ocr, scrape, transcribe, vision


@pytest.fixture
def gpu_bf16(monkeypatch):
    monkeypatch.setattr(transcribe, "cuda_usable", lambda: True)
    monkeypatch.setattr(transcribe, "has_bfloat16_support", lambda: True)


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(transcribe, "cuda_usable", lambda: False)
    monkeypatch.setattr(transcribe, "has_bfloat16_support", lambda: False)
    monkeypatch.setattr(utilities, "cuda_usable", lambda: False)


@pytest.fixture
def scrape_settings(monkeypatch):
    app = f"ScrapeDocumentation-pytest-{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(scrape, "QSETTINGS_APP", app)
    yield app
    QSettings(scrape.QSETTINGS_ORG, app).clear()


def test_whisper_table_groups_precisions(gpu_bf16):
    table = transcribe.model_table()
    assert len(table) == 10
    assert all(set(entry) == {"float32", "bfloat16", "float16"} for entry in table.values())
    assert all(p["available"] for entry in table.values() for p in entry.values())
    assert table["Whisper large-v3"]["float16"]["key"] == "Whisper large-v3 - float16"


def test_whisper_table_on_cpu_only_offers_float32(cpu_only):
    table = transcribe.model_table()
    for entry in table.values():
        assert entry["float32"]["available"]
        assert not entry["bfloat16"]["available"] and not entry["float16"]["available"]


def test_transcribe_model_and_precision(gpu_bf16):
    tool = transcribe.TranscribeTool(None)
    assert tool.model == "Distil Whisper large-v3.5" and tool.precision == "float32"
    tool.set_precision("bfloat16")
    tool.set_model("Whisper large-v3")
    assert tool.model_key() == "Whisper large-v3 - bfloat16"
    tool.set_model("not a model")
    assert tool.model == "Whisper large-v3"
    state = tool.state()
    assert [p["value"] for p in state["precisions"]] == ["float32", "bfloat16", "float16"]
    assert state["running"] is False and state["file"] is None


def test_transcribe_precision_falls_back_when_unavailable(gpu_bf16, monkeypatch):
    tool = transcribe.TranscribeTool(None)
    tool.set_precision("float16")
    tool.table["Whisper base.en"]["float16"]["available"] = False
    tool.set_model("Whisper base.en")
    assert tool.precision == "float32"
    tool.set_precision("float16")
    assert tool.precision == "float32"


def test_transcribe_batch_validation(cpu_only):
    tool = transcribe.TranscribeTool(None)
    assert tool.set_batch("abc") and tool.batch == 8
    assert tool.set_batch(0) and tool.set_batch(151)
    assert tool.set_batch(" 32 ") is None and tool.batch == 32


def test_transcribe_start_needs_a_file_and_no_build(cpu_only, monkeypatch, tmp_path):
    tool = transcribe.TranscribeTool(None)
    tool.start()
    assert tool.result == {"ok": False, "message": "Choose an audio file first."}
    tool.file = str(tmp_path / "talk.mp3")
    monkeypatch.setattr(transcribe, "database_build_running", lambda: True)
    tool.start()
    assert tool.result["message"] == transcribe.BUILD_RUNNING_MESSAGE and tool.worker is None


def test_transcript_name_finds_numbered_copies(cpu_only, monkeypatch, tmp_path):
    monkeypatch.setattr(transcribe, "PROJECT_ROOT", tmp_path)
    docs = tmp_path / "Docs_for_DB"
    docs.mkdir()
    tool = transcribe.TranscribeTool(None)
    tool.file = str(tmp_path / "meeting [draft].mp3")
    tool.started = time.time()
    (docs / "meeting [draft].json").write_text("{}", encoding="utf-8")
    time.sleep(0.05)
    (docs / "meeting [draft] (2).json").write_text("{}", encoding="utf-8")
    (docs / "other.json").write_text("{}", encoding="utf-8")
    assert tool._transcript_name() == "meeting [draft] (2).json"


def test_scrape_state_marks_scraped_docs(monkeypatch, tmp_path, scrape_settings):
    monkeypatch.setattr(scrape, "PROJECT_ROOT", tmp_path)
    folder = scrape.scrape_documentation["aiohttp"]["folder"]
    (tmp_path / "Scraped_Documentation" / folder).mkdir(parents=True)
    tool = scrape.ScrapeTool(None)
    state = tool.state()
    assert len(state["docs"]) == len(scrape.scrape_documentation)
    assert [d["name"] for d in state["docs"] if d["scraped"]] == ["aiohttp"]
    assert state["selected"] == state["docs"][0]["name"] and state["limit"] == 6 and state["rows"] == []
    tool.select("aiohttp")
    assert tool.selected == "aiohttp"
    tool.select("nope")
    assert tool.selected == "aiohttp"


def test_rate_limited_scrapes_survive_a_restart(monkeypatch, tmp_path, scrape_settings):
    monkeypatch.setattr(scrape, "PROJECT_ROOT", tmp_path)
    folder = tmp_path / "Scraped_Documentation" / scrape.scrape_documentation["anyio"]["folder"]
    folder.mkdir(parents=True)
    for i in range(3):
        (folder / f"p{i}.html").write_text("x", encoding="utf-8")
    scrape._mark_rate_limited_persistent("anyio")
    scrape._mark_rate_limited_persistent("aiohttp")
    tool = scrape.ScrapeTool(None)
    assert tool.rows == {"anyio": {"status": "rate_limited", "pages": 3, "folder": str(folder)}}
    assert scrape._load_rate_limited_set() == {"anyio"}
    tool.dismiss("anyio")
    assert tool.rows == {} and scrape._load_rate_limited_set() == set()


def test_scrape_finish_updates_rows(monkeypatch, tmp_path, scrape_settings):
    monkeypatch.setattr(scrape, "PROJECT_ROOT", tmp_path)
    tool = scrape.ScrapeTool(None)
    folder = tmp_path / "Scraped_Documentation" / "x"
    folder.mkdir(parents=True)
    (folder / "a.html").write_text("x", encoding="utf-8")
    tool.rows["attrs"] = {"status": "scraping", "pages": 0, "folder": str(folder)}
    tool._on_worker_finished("attrs", False, True)
    assert tool.rows["attrs"]["status"] == "rate_limited" and tool.rows["attrs"]["pages"] == 1
    assert "attrs" in scrape._load_rate_limited_set()
    tool._on_worker_finished("attrs", False, False)
    assert tool.rows["attrs"]["status"] == "completed" and "attrs" not in scrape._load_rate_limited_set()
    assert tool.selected == "attrs"


def test_vision_counts_images_and_reads_the_chosen_model(cpu_only, monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "Docs_for_DB"
    docs.mkdir()
    for name in ("a.png", "b.JPG", "c.pdf", "d.txt"):
        (docs / name).write_bytes(b"x")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"vision": {"chosen_model": "Qwen VL - 7b"}}), encoding="utf-8")
    tool = vision.VisionTool(None)
    assert tool.images == 2
    assert tool.chosen == "Liquid-VL - 480M"
    assert tool.selected == ["Liquid-VL - 480M"]
    tool.set_selected(["Qwen VL - 7b", "Liquid-VL - 480M"])
    assert tool.selected == ["Liquid-VL - 480M"]


def test_vision_guards(cpu_only, monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    tool = vision.VisionTool(None)
    tool.summarize()
    assert tool.result["ok"] is False and "Create Database tab" in tool.result["message"]
    tool.compare()
    assert tool.result["message"] == "Choose an image first."
    tool.file = str(tmp_path / "x.png")
    tool.set_selected([])
    tool.compare()
    assert tool.result["message"] == "Please select at least one model."


def test_vision_result_files(tmp_path):
    documents = [
        {"page_content": "one two", "metadata": {"source": "a.png"}},
        {"page_content": "three four five six", "metadata": {"file_path": "b.png"}},
    ]
    contents, average = vision.extract_page_content(documents)
    assert [c[0] for c in contents] == ["a.png", "b.png"] and average == 13
    text = open(vision.save_page_contents(contents, average), encoding="utf-8").read()
    assert "Average Summary Length: 13.00 characters" in text and "File Path: b.png" in text
    path = vision.save_comparison_results("img.png", [("Model A", "abc" * 10, 2.0), ("Model B", "Error processing with Model B", 0.0)])
    table = open(path, encoding="utf-8").read()
    assert "Image Path: img.png" in table and "|Model A" in table and "15.0" in table


def test_ocr_summaries_and_output(tmp_path):
    warnings, infos = ocr.summarize_events({
        "lowconf": [{"page": 2}, {"page": 2}],
        "notext": [{"page": 4, "ink_frac": 0.01}, {"page": 5, "ink_frac": 0.0}],
        "oriented": [{"page": 1}],
        "fileerror": [{"error": "bad"}],
    })
    assert warnings == [
        "1 low-confidence page(s): 2 (worth a manual review)",
        "1 page(s) with visible content but no OCR text: 4",
        "file failed: bad",
    ]
    assert infos == ["1 blank page(s): 5", "1 page(s) auto-rotated to read: 1"]
    assert ocr.format_seconds(75.25) == "1m 15.2s" and ocr.format_seconds(3.04) == "3.0s"
    tool = ocr.OcrTool(None)
    assert tool.output_path() is None
    tool.file = str(tmp_path / "scan.pdf")
    assert tool.output_path() == tmp_path / "scan_OCR.pdf"
    tool.set_engine("tesseract")
    tool.set_engine("nope")
    assert tool.engine == "tesseract"
    tool.file = None
    tool.start()
    assert tool.result == {"ok": False, "message": "Choose a PDF file first."}


def test_ocr_process_documents_reports_progress(monkeypatch):
    import modules.ocr as ocr_module

    class FakeProcess:
        def __init__(self, target, args):
            self.progress_queue = args[-1]
            self.exitcode = 0

        def start(self):
            for message in (("total", 3), ("update", 1), ("update", 2), ("lowconf", {"page": 1}), ("done", None)):
                self.progress_queue.put(message)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    monkeypatch.setattr(ocr_module, "Process", FakeProcess)
    monkeypatch.setattr(ocr_module, "Queue", queue_module.Queue)
    monkeypatch.setattr(ocr_module.time, "sleep", lambda seconds: None)
    seen = []
    events = ocr_module.process_documents(pdf_paths=__import__("pathlib").Path("x.pdf"), backend="rapidocr", on_progress=lambda *a: seen.append(a))
    assert seen == [("total", 3), ("update", 1), ("update", 2)]
    assert events["lowconf"] == [{"page": 1}]
    assert ocr_module.process_documents.__defaults__[-1] is None


def test_backup_summary(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("alpha", "beta", "user_manual"):
        (tmp_path / "Vector_DB" / name).mkdir(parents=True)
    (tmp_path / "Vector_DB" / "notes.txt").write_text("x", encoding="utf-8")
    tool = backups.BackupTool(None)
    assert tool.state() == {"databases": 2, "missing": ["alpha", "beta"], "has_backup": False,
                            "task": None, "started": None, "result": None}
    (tmp_path / "Vector_DB_Backup" / "alpha").mkdir(parents=True)
    tool.refresh()
    assert tool.state()["missing"] == ["beta"] and tool.state()["has_backup"] is True
    assert tool.busy_message() is None

    class Running:
        def isRunning(self):
            return True

    tool.worker, tool.task = Running(), "restore"
    assert tool.busy_message() == "A database restore is still running. Please wait for it to finish before closing the program."


def test_gpu_rows_filter_and_sort(monkeypatch):
    monkeypatch.setattr(gpus, "GPUS", {
        "Small": {"memory_size_gb": 8, "memory_type": "GDDR6", "cuda_cores": 3000, "tensor_cores": 96, "architecture": "Ampere",
                  "cuda_major_version": 8, "cuda_minor_version": 6, "half_float_performance_gflop_s": 12000, "release_date": date(2021, 1, 1)},
        "Big": {"memory_size_gb": 16, "memory_type": "GDDR6X", "cuda_cores": 9000, "tensor_cores": 288, "architecture": "Ada Lovelace",
                "cuda_major_version": 8, "cuda_minor_version": 9, "half_float_performance_gflop_s": 40000, "release_date": None},
        "Huge": {"memory_size_gb": 24, "memory_type": "GDDR6X", "cuda_cores": 16384, "tensor_cores": 512, "architecture": "Ada Lovelace",
                 "cuda_major_version": 8, "cuda_minor_version": 9, "half_float_performance_gflop_s": 82000, "release_date": date(2022, 10, 12)},
    })
    rows = gpus.gpu_rows(8, 16)
    assert [r["name"] for r in rows] == ["Big", "Small"]
    assert rows[0]["cc"] == "8.9" and rows[0]["released"] is None and rows[1]["released"] == 2021 and rows[0]["fp16"] == 40.0
    assert gpus.gpu_rows(17, 23) == []


def test_real_gpu_table_sizes():
    assert gpus.VRAM_SIZES == sorted(set(gpus.VRAM_SIZES)) and 8 in gpus.VRAM_SIZES and 16 in gpus.VRAM_SIZES
    assert all(r["cores"] > 0 for r in gpus.gpu_rows(1, 200))
