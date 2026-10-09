import json
import os

import pytest
import yaml

import core.utilities as utilities
from gui.tabs_databases import create_build, create_config, create_files


@pytest.fixture
def root(tmp_path, monkeypatch):
    for module in (create_build, create_config, create_files):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "Docs_for_DB").mkdir()
    (tmp_path / "Vector_DB").mkdir()
    return tmp_path


def write_config(root, data):
    (root / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def read_config(root):
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def test_file_kinds():
    assert create_files.kind_of("Report.PDF") == "pdf"
    assert create_files.kind_of("notes.md") == "text"
    assert create_files.kind_of("sheet.xlsm") == "sheet"
    assert create_files.kind_of("call.json") == "transcript"
    assert create_files.kind_of("photo.TIFF") == "image"
    assert create_files.kind_of("archive.zip") == "other"


def test_staged_files_list_summary_and_remove(root):
    docs = root / "Docs_for_DB"
    (docs / "b.pdf").write_text("x", encoding="utf-8")
    (docs / "a.txt").write_text("x", encoding="utf-8")
    (docs / "talk.json").write_text("{}", encoding="utf-8")
    (docs / "subfolder").mkdir()
    files = create_files.StagedFiles(None)
    assert files.names == ["a.txt", "b.pdf", "talk.json"]
    assert files.summary() == {"text": 1, "pdf": 1, "transcript": 1}
    rows = files.entries()
    assert [r["kind"] for r in rows] == ["text", "pdf", "transcript"] and rows[0]["target"].endswith("a.txt")
    version = files.version
    assert files.remove(["a.txt", "../outside.txt", "missing.pdf"]) == 1
    assert files.names == ["b.pdf", "talk.json"] and files.version == version + 1
    files.refresh()
    assert files.version == version + 1


def test_staged_files_keep_symlink_targets(root, tmp_path_factory):
    source = tmp_path_factory.mktemp("source") / "contract.pdf"
    source.write_text("x", encoding="utf-8")
    link = root / "Docs_for_DB" / "contract.pdf"
    try:
        os.symlink(source, link)
    except OSError:
        pytest.skip("symlinks are not available")
    files = create_files.StagedFiles(None)
    assert files.entries()[0]["target"] == str(source)
    files.remove(["contract.pdf"])
    assert not link.exists() and source.exists()


def test_available_models_hide_incomplete_and_gpu_only(tmp_path, monkeypatch):
    models = tmp_path / "vector"
    for name in ("BAAI--bge-small-en-v1.5", "Qwen--Qwen3-Embedding-8B", "my-custom-model", "half-done"):
        (models / name).mkdir(parents=True)
    (models / "half-done" / ".download_incomplete").write_text("", encoding="utf-8")
    monkeypatch.setattr(create_config, "models_dir", lambda: models)
    monkeypatch.setattr(utilities, "cuda_usable", lambda: False)
    rows = create_config.available_models()
    assert [r["folder"] for r in rows] == ["BAAI--bge-small-en-v1.5", "my-custom-model"]
    bge, custom = rows
    assert bge["name"] == "bge-small-en-v1.5" and bge["dimensions"] == 384 and bge["known"]
    assert custom["name"] == "my-custom-model" and not custom["known"]
    monkeypatch.setattr(utilities, "cuda_usable", lambda: True)
    assert "Qwen--Qwen3-Embedding-8B" in [r["folder"] for r in create_config.available_models()]


def test_embedding_dimensions(tmp_path):
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / "config.json").write_text(json.dumps({"hidden_size": 768}), encoding="utf-8")
    d_model = tmp_path / "t5"
    d_model.mkdir()
    (d_model / "config.json").write_text(json.dumps({"d_model": 512}), encoding="utf-8")
    assert create_config.embedding_dimensions(str(hidden)) == 768
    assert create_config.embedding_dimensions(str(d_model)) == 512
    assert create_config.embedding_dimensions(str(tmp_path / "stella_en_400M")) == 1024
    assert create_config.embedding_dimensions(str(tmp_path / "missing")) is None


def test_select_model_saves_and_clears(root):
    model = root / "Models" / "vector" / "acme--embedder"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"hidden_size": 1024}), encoding="utf-8")
    write_config(root, {"database": {"chunk_size": 700}})
    create_config.select_model(str(model))
    cfg = read_config(root)
    assert cfg["EMBEDDING_MODEL_NAME"] == str(model) and cfg["EMBEDDING_MODEL_DIMENSIONS"] == 1024
    assert cfg["database"]["chunk_size"] == 700
    create_config.select_model(None)
    assert "EMBEDDING_MODEL_NAME" not in read_config(root) and "EMBEDDING_MODEL_DIMENSIONS" not in read_config(root)


def test_select_model_never_overwrites_an_unreadable_config(root):
    (root / "config.yaml").write_text("database: [unclosed", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        create_config.select_model(str(root / "anything"))
    assert (root / "config.yaml").read_text(encoding="utf-8") == "database: [unclosed"


def test_settings_summary_and_existing_names(root, monkeypatch):
    monkeypatch.setattr(create_config, "cuda_usable", lambda: True)
    (root / "Vector_DB" / "contracts").mkdir()
    (root / "Vector_DB" / "stray.txt").write_text("x", encoding="utf-8")
    cfg = {
        "EMBEDDING_MODEL_NAME": str(root / "Models" / "vector" / "BAAI--bge-small-en-v1.5"),
        "database": {"chunk_size": 700, "chunk_overlap": 250, "half": True, "pipeline_preset": "high"},
        "Compute_Device": {"available": ["cpu", "cuda"], "database_creation": "cpu"},
        "vision": {"chosen_model": "Liquid-VL - 480M"},
        "created_databases": {"stale_entry": {}},
    }
    summary = create_config.settings_summary(cfg)
    assert summary["chunk_size"] == 700 and summary["chunk_overlap"] == 250 and summary["preset"] == "high"
    assert summary["device"] == "cpu" and summary["cpu_warning"] is True and summary["precision"] == "float32"
    assert summary["vision"] == "Liquid-VL - 480M" and summary["vision_needs_gpu"] is False
    assert create_config.existing_databases() == ["contracts"]
    assert create_config.name_limit() >= 3


def test_build_progress_follows_the_real_build_log():
    progress = create_build.BuildProgress()
    lines = [
        "Initializing database creation...",
        "\x1b[33mExtracting documents (subprocess)...\x1b[0m",
        "2026-10-07 17:34:52,100 INFO [db.database_interactions] Extracted 2 documents",
        "Processing any audio transcripts...",
        "Processing any images...",
        "\x1b[33mSplitting documents into chunks (subprocess)...\x1b[0m",
        "2026-10-07 17:34:58,000 INFO [db.database_interactions] Split into 1,209 chunks",
        "\x1b[33m\nComputing vectors...\x1b[0m",
        "2026-10-07 17:34:58,700 INFO [db.embedding_models]   [Tokenize (attempt 1)] 2026-10-07 17:34:58,700 INFO [stage_tokenize] Stage 3: Tokenizing (subprocess-per-chunk isolation)",
        "2026-10-07 17:35:09,860 INFO [db.embedding_models] Tokenization complete: 1200 batches, 0 errors, 99.0% padding efficiency",
    ]
    for line in lines:
        progress.feed(line)
    state = progress.state()
    assert state["stage"] == "tokenize" and state["batches"] is None
    assert [s["status"] for s in state["stages"]] == ["done", "done", "done", "running", "pending", "pending"]
    for line in [
        "2026-10-07 17:35:09,873 INFO [db.embedding_models] Running forward pass on 1200 pre-padded batches...",
        "2026-10-07 17:35:10,000 INFO [db.embedding_models]   Forward pass: 500/1200 batches",
    ]:
        progress.feed(line)
    state = progress.state()
    assert state["stage"] == "embed" and state["documents"] == 2 and state["chunks"] == 1209
    assert (state["batch"], state["batches"]) == (500, 1200)
    assert [s["status"] for s in state["stages"]] == ["done", "done", "done", "done", "running", "pending"]
    progress.feed("Processing any images...")
    assert progress.stage == "embed"
    for line in [
        "2026-10-07 17:35:10,036 INFO [db.embedding_models] Forward pass complete: 1200 batches processed",
        "2026-10-07 17:35:10,469 INFO [db.database_interactions] Write attempt 1/5",
        "\x1b[32mDatabase created. Total time: 20.46 seconds.\x1b[0m",
    ]:
        progress.feed(line)
    state = progress.state()
    assert state["batch"] == 1200 and state["stage"] == "done"
    assert all(s["status"] == "done" for s in state["stages"])
    assert "\x1b" not in "".join(progress.log) and progress.feed("   ") is False


def test_update_config_records_the_new_database(root):
    write_config(root, {"EMBEDDING_MODEL_NAME": "D:/models/bge", "database": {"chunk_size": 900, "chunk_overlap": 100}})
    create_build.update_config_with_database_name("new_db")
    assert read_config(root)["created_databases"]["new_db"] == {"model": "D:/models/bge", "chunk_size": 900, "chunk_overlap": 100}


def test_cpu_confirmation(monkeypatch):
    asked = []
    monkeypatch.setattr(create_build.QMessageBox, "question", staticmethod(lambda *a, **k: asked.append(a) or create_build.QMessageBox.No))
    assert create_build.confirm_cpu_creation(None, {"Compute_Device": {"available": ["cpu", "cuda"], "database_creation": "cuda"}})
    assert create_build.confirm_cpu_creation(None, {"Compute_Device": {"available": ["cpu"], "database_creation": "cpu"}})
    assert not asked
    assert not create_build.confirm_cpu_creation(None, {"Compute_Device": {"available": ["cpu", "cuda"], "database_creation": "cpu"}})
    assert len(asked) == 1


def test_build_start_guards(root, monkeypatch):
    monkeypatch.setattr(create_build, "transcription_running", lambda: False)
    files = create_files.StagedFiles(None)
    controller = create_build.BuildController(None, files)
    cfg = {"EMBEDDING_MODEL_NAME": str(root / "model")}
    assert controller.start(None, "good_name", "model", True, {})["error"] == "Please select a model before creating a database."
    assert controller.start(None, "  ", "model", True, cfg)["error"] == "Please enter a database name before creating a database."
    assert "empty" in controller.start(None, "good_name", "model", True, cfg)["error"]
    assert controller.result["kind"] == "error" and controller.phase == "idle"
    monkeypatch.setattr(create_build, "transcription_running", lambda: True)
    assert "transcription" in controller.start(None, "good_name", "model", True, cfg)["error"]
