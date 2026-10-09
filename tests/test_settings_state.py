import os
import shutil

import pytest
import yaml

import core.utilities as utilities
from core.constants import PROJECT_ROOT
from gui.tabs_settings import settings_state as ss


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    shutil.copy(PROJECT_ROOT / "Assets" / "config.yaml", tmp_path / "config.yaml")
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")) or {}
    cfg["Compute_Device"] = {"available": ["cpu", "cuda"], "database_creation": "cuda", "database_query": "cpu"}
    cfg.setdefault("database", {}).update({"chunk_size": 700, "chunk_overlap": 250, "similarity": 0.5, "contexts": 5})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(utilities, "cuda_usable", lambda: True)
    monkeypatch.setattr(ss, "cuda_usable", lambda: True)
    monkeypatch.setattr(ss, "gpu_name", lambda: "Test GPU")


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(utilities, "cuda_usable", lambda: False)
    monkeypatch.setattr(ss, "cuda_usable", lambda: False)


def config(folder):
    return yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))


def test_build_state_reads_the_saved_values(sandbox, gpu):
    state = ss.build_state()
    assert state["error"] is None
    assert state["values"]["query.similarity"] == 0.5
    assert state["values"]["create.chunk_size"] == 700
    assert state["values"]["query.device"] == "cpu"
    assert [s["id"] for s in state["sections"]] == ["query", "create", "tts", "vision"]


def test_number_settings_save_and_validate(sandbox, gpu):
    assert ss.apply_change("query.similarity", 0.65)["ok"]
    assert ss.apply_change("query.contexts", "1,000")["ok"]
    cfg = config(sandbox)["database"]
    assert cfg["similarity"] == 0.65 and cfg["contexts"] == 1000

    result = ss.apply_change("query.similarity", "1.5")
    assert not result["ok"] and result["error"] == ss.MESSAGES["query.similarity"]
    assert not ss.apply_change("query.contexts", "abc")["ok"]
    assert not ss.apply_change("query.contexts", 1001)["ok"]
    assert config(sandbox)["database"]["similarity"] == 0.65


def test_unchanged_values_do_not_rewrite_the_file(sandbox, gpu):
    before = os.stat(sandbox / "config.yaml").st_mtime_ns
    result = ss.apply_change("query.similarity", 0.5)
    assert result["ok"] and result["changed"] is False
    assert os.stat(sandbox / "config.yaml").st_mtime_ns == before


def test_chunk_overlap_must_stay_below_chunk_size(sandbox, gpu):
    result = ss.apply_change("create.chunk_size", 200)
    assert not result["ok"] and "250" in result["error"]
    assert not ss.apply_change("create.chunk_overlap", 700)["ok"]
    assert ss.apply_change("create.chunk_overlap", 100)["ok"]
    assert ss.apply_change("create.chunk_size", 200)["ok"]
    cfg = config(sandbox)["database"]
    assert (cfg["chunk_size"], cfg["chunk_overlap"]) == (200, 100)


def test_text_and_choice_settings(sandbox, gpu):
    assert ss.apply_change("query.search_term", "  lease  ")["ok"]
    assert config(sandbox)["database"]["search_term"] == "lease"
    assert ss.apply_change("query.search_term", "")["ok"]
    assert config(sandbox)["database"]["search_term"] == ""
    assert ss.apply_change("query.document_types", "audio")["ok"]
    assert not ss.apply_change("query.document_types", "video")["ok"]
    assert ss.apply_change("create.pipeline_preset", "maximum")["ok"]
    assert not ss.apply_change("create.pipeline_preset", "turbo")["ok"]
    assert ss.apply_change("query.device", "cuda")["ok"]
    assert not ss.apply_change("query.device", "mps")["ok"]
    cfg = config(sandbox)
    assert cfg["database"]["document_types"] == "audio"
    assert cfg["database"]["pipeline_preset"] == "maximum"
    assert cfg["Compute_Device"]["database_query"] == "cuda"


def test_half_precision_needs_a_gpu(sandbox, cpu):
    result = ss.apply_change("create.half", True)
    assert not result["ok"] and "NVIDIA" in result["error"]
    assert ss.build_state()["values"]["create.half"] is False


def test_tts_backends_write_the_same_keys_as_before(sandbox, gpu):
    assert ss.apply_change("tts.backend", "kyutai")["ok"]
    assert ss.apply_change("tts.option", {"backend": "kyutai", "name": "model", "value": "0.75B (EN, ~2GB VRAM)"})["ok"]
    kyutai = config(sandbox)["kyutai"]
    assert kyutai["hf_repo"] == "kyutai/tts-0.75b-en-public" and kyutai["n_q"] == 16
    assert kyutai["temp"] == 0.6 and kyutai["cfg_coef"] == 2.0

    assert ss.apply_change("tts.backend", "kokoro")["ok"]
    assert ss.apply_change("tts.option", {"backend": "kokoro", "name": "speed", "value": "Fast"})["ok"]
    assert config(sandbox)["kokoro"]["speed"] == 1.6

    assert ss.apply_change("tts.backend", "whisperspeech")["ok"]
    assert ss.apply_change("tts.option", {"backend": "whisperspeech", "name": "t2s", "value": "t2s-small"})["ok"]
    assert config(sandbox)["tts"]["t2s"] == "t2s-small-en+pl.model"
    refused = ss.apply_change("tts.option", {"backend": "whisperspeech", "name": "speaker", "value": ss.WHISPER_SPEECH_VOICE_CLONING_LABEL})
    assert not refused["ok"]
    assert not ss.apply_change("tts.option", {"backend": "kokoro", "name": "voice", "value": "af"})["ok"]


def test_gpu_only_choices_are_refused_on_a_cpu(sandbox, cpu):
    assert "bark" not in ss.available_backends()
    assert not ss.apply_change("tts.backend", "bark")["ok"]
    assert ss.available_vision_models() == ["Liquid-VL - 480M"]
    assert not ss.apply_change("vision.model", "Qwen VL - 7b")["ok"]
    assert ss.build_state()["sections"][1]["fields"][3]["disabled"] is True


def test_normalize_writes_only_when_something_changes(sandbox, gpu):
    cfg = config(sandbox)
    cfg.pop("vision", None)
    (sandbox / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert ss.normalize_config() is True
    assert config(sandbox)["vision"]["chosen_model"] == ss.DEFAULT_VISION_MODEL
    before = os.stat(sandbox / "config.yaml").st_mtime_ns
    assert ss.normalize_config() is False
    assert os.stat(sandbox / "config.yaml").st_mtime_ns == before


def test_normalize_falls_back_from_gpu_only_backends(sandbox, cpu):
    cfg = config(sandbox)
    cfg.setdefault("tts", {})["model"] = "bark"
    (sandbox / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert ss.normalize_config() is True
    assert config(sandbox)["tts"]["model"] == "googletts"


def test_an_unreadable_config_is_reported_not_overwritten(sandbox, gpu):
    (sandbox / "config.yaml").write_text("database: [unclosed", encoding="utf-8")
    assert ss.build_state()["error"]
    assert ss.normalize_config() is False
    assert not ss.apply_change("query.similarity", 0.4)["ok"]
    assert (sandbox / "config.yaml").read_text(encoding="utf-8") == "database: [unclosed"
