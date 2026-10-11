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
    monkeypatch.setattr(ss.torch.cuda, "get_device_capability", lambda *args: (8, 9))
    monkeypatch.setattr(utilities, "has_bfloat16_support", lambda: True)


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(utilities, "cuda_usable", lambda: False)
    monkeypatch.setattr(ss, "cuda_usable", lambda: False)


def config(folder):
    return yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))


def save(folder, cfg):
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def tts_backend(key):
    return next(b for b in ss.build_state()["tts"]["backends"] if b["key"] == key)


def benchmark(backend, variant, run):
    return ss.TTS_BENCHMARKS[backend][variant][run]


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
    assert ss.apply_change("tts.option", {"backend": "kyutai", "name": "model", "value": "0.75B (English)"})["ok"]
    kyutai = config(sandbox)["kyutai"]
    assert kyutai["hf_repo"] == "kyutai/tts-0.75b-en-public" and kyutai["n_q"] == 16
    assert kyutai["temp"] == 0.6 and kyutai["cfg_coef"] == 2.0

    assert ss.apply_change("tts.backend", "kokoro")["ok"]
    assert "speed" not in config(sandbox)["kokoro"]
    assert not ss.apply_change("tts.option", {"backend": "kokoro", "name": "speed", "value": "Fast"})["ok"]

    assert ss.apply_change("tts.backend", "whisperspeech")["ok"]
    assert ss.apply_change("tts.option", {"backend": "whisperspeech", "name": "t2s", "value": "t2s-small"})["ok"]
    assert config(sandbox)["tts"]["t2s"] == "t2s-small-en+pl.model"
    refused = ss.apply_change("tts.option", {"backend": "whisperspeech", "name": "speaker", "value": ss.WHISPER_SPEECH_VOICE_CLONING_LABEL})
    assert not refused["ok"]
    assert not ss.apply_change("tts.option", {"backend": "kokoro", "name": "voice", "value": "af"})["ok"]


def test_kyutai_pocket_drops_the_old_int8_setting(sandbox, gpu):
    cfg = config(sandbox)
    cfg["kyutaipocket"] = {"voice": "alba", "quantize": True}
    (sandbox / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert ss.apply_change("tts.backend", "kyutaipocket")["ok"]
    assert "quantize" not in config(sandbox)["kyutaipocket"]
    assert "toggles" not in tts_backend("kyutaipocket")
    assert not ss.apply_change("tts.option", {"backend": "kyutaipocket", "name": "quantize", "value": True})["ok"]


def test_gpu_only_choices_are_refused_on_a_cpu(sandbox, cpu):
    assert "bark" not in ss.available_backends()
    assert not ss.apply_change("tts.backend", "bark")["ok"]
    assert ss.available_vision_models() == ["Liquid-VL - 480M"]
    assert not ss.apply_change("vision.model", "Qwen VL - 7b")["ok"]
    assert ss.build_state()["sections"][1]["fields"][3]["disabled"] is True


def test_only_google_kokoro_and_pocket_run_without_a_gpu(sandbox, cpu):
    assert ss.available_backends() == ["googletts", "kokoro", "kyutaipocket"]
    assert not ss.apply_change("tts.backend", "chattts")["ok"]
    cfg = config(sandbox)
    cfg.setdefault("tts", {})["model"] = "chattts"
    (sandbox / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert ss.normalize_config() is True
    assert config(sandbox)["tts"]["model"] == "googletts"


def test_kokoro_runs_on_the_cpu_until_the_gpu_is_chosen(sandbox, gpu):
    kokoro = tts_backend("kokoro")
    assert kokoro["device"]["value"] == "cpu"
    assert kokoro["precision"]["disabled"] and kokoro["precision"]["value"] == "float32"
    assert (kokoro["card"]["device"], kokoro["card"]["dtype"]) == ("cpu", "float32")
    assert kokoro["card"]["benchmark"] == benchmark("kokoro", "", "cpu-float32")

    assert ss.apply_change("tts.option", {"backend": "kokoro", "name": "device", "value": "cuda"})["ok"]
    kokoro = tts_backend("kokoro")
    assert kokoro["precision"]["value"] == "half" and not kokoro["precision"]["disabled"]
    assert kokoro["card"]["dtype"] == "bfloat16"
    assert kokoro["card"]["benchmark"] == benchmark("kokoro", "", "cuda-bfloat16")

    assert ss.apply_change("tts.option", {"backend": "kokoro", "name": "precision", "value": "float32"})["ok"]
    assert tts_backend("kokoro")["card"]["benchmark"] == benchmark("kokoro", "", "cuda-float32")
    saved = config(sandbox)["kokoro"]
    assert (saved["device"], saved["precision"]) == ("cuda", "float32")


def test_kyutai_pocket_always_runs_in_half_precision_on_the_gpu(sandbox, gpu):
    assert ss.apply_change("tts.backend", "kyutaipocket")["ok"]
    pocket = tts_backend("kyutaipocket")
    assert pocket["device"]["value"] == "cuda" and pocket["precision"] is None
    assert pocket["card"]["dtype"] == "bfloat16"
    assert pocket["card"]["benchmark"] == benchmark("kyutaipocket", "", "cuda-bfloat16")
    saved = config(sandbox)["kyutaipocket"]
    assert saved["device"] == "cuda" and "precision" not in saved
    assert not ss.apply_change("tts.option", {"backend": "kyutaipocket", "name": "precision", "value": "float32"})["ok"]

    assert ss.apply_change("tts.option", {"backend": "kyutaipocket", "name": "device", "value": "cpu"})["ok"]
    assert tts_backend("kyutaipocket")["card"]["benchmark"] == benchmark("kyutaipocket", "", "cpu-float32")


def test_pocket_and_kyutai_ignore_a_saved_float32_choice(sandbox, gpu):
    cfg = config(sandbox)
    cfg["kyutaipocket"] = {"device": "cuda", "precision": "float32", "voice": "alba"}
    cfg["kyutai"] = {"precision": "float32"}
    save(sandbox, cfg)
    assert tts_backend("kyutaipocket")["card"]["dtype"] == "bfloat16"
    assert tts_backend("kyutai")["card"]["dtype"] == "bfloat16"
    assert ss.apply_change("tts.backend", "kyutaipocket")["ok"]
    assert "precision" not in config(sandbox)["kyutaipocket"]


@pytest.mark.parametrize("backend", ["chattts", "chatterbox"])
def test_gpu_only_backends_choose_between_half_and_float32(sandbox, gpu, backend):
    assert ss.apply_change("tts.backend", backend)["ok"]
    state = tts_backend(backend)
    assert state["device"] is None and state["precision"]["value"] == "half"
    assert state["card"]["benchmark"] == benchmark(backend, "", "cuda-bfloat16")
    assert config(sandbox)[backend]["precision"] == "half"
    assert ss.apply_change("tts.option", {"backend": backend, "name": "precision", "value": "float32"})["ok"]
    assert tts_backend(backend)["card"]["benchmark"] == benchmark(backend, "", "cuda-float32")
    assert not ss.apply_change("tts.option", {"backend": backend, "name": "device", "value": "cpu"})["ok"]


def test_kyutai_cards_follow_the_model(sandbox, gpu):
    assert ss.apply_change("tts.backend", "kyutai")["ok"]
    kyutai = tts_backend("kyutai")
    assert kyutai["device"] is None and kyutai["precision"] is None
    assert kyutai["card"]["name"] == "Kyutai 1.6B"
    assert kyutai["card"]["benchmark"] == benchmark("kyutai", "1.6B", "cuda-bfloat16")
    assert ss.apply_change("tts.option", {"backend": "kyutai", "name": "model", "value": "0.75B (English)"})["ok"]
    kyutai = tts_backend("kyutai")
    assert kyutai["card"]["name"] == "Kyutai 0.75B"
    assert kyutai["card"]["url"] == "https://huggingface.co/kyutai/tts-0.75b-en-public"
    assert kyutai["card"]["benchmark"] == benchmark("kyutai", "0.75B", "cuda-bfloat16")
    assert not ss.apply_change("tts.option", {"backend": "kyutai", "name": "precision", "value": "float32"})["ok"]


def test_bark_chooses_between_half_and_float32(sandbox, gpu):
    assert ss.apply_change("tts.backend", "bark")["ok"]
    bark = tts_backend("bark")
    assert bark["device"] is None and bark["precision"]["value"] == "half"
    assert bark["card"]["dtype"] == "float16" and bark["precision"]["hint"] == "Runs in float16 on this GPU"
    assert bark["card"]["benchmark"] == benchmark("bark", "small", "cuda-float16")
    assert config(sandbox)["bark"]["precision"] == "half"
    assert ss.apply_change("tts.option", {"backend": "bark", "name": "precision", "value": "float32"})["ok"]
    assert ss.apply_change("tts.option", {"backend": "bark", "name": "size", "value": "normal"})["ok"]
    bark = tts_backend("bark")
    assert bark["card"]["name"] == "Bark" and bark["card"]["dtype"] == "float32"
    assert bark["card"]["benchmark"] == benchmark("bark", "normal", "cuda-float32")


def test_whisperspeech_stays_in_float16(sandbox, gpu):
    assert ss.apply_change("tts.backend", "whisperspeech")["ok"]
    whisper = tts_backend("whisperspeech")
    assert whisper["device"] is None and whisper["precision"] is None and whisper["card"]["dtype"] == "float16"
    assert ss.apply_change("tts.option", {"backend": "whisperspeech", "name": "t2s", "value": "t2s-small"})["ok"]
    s2a = ss.tts_values(config(sandbox))["whisperspeech"]["s2a"]
    assert tts_backend("whisperspeech")["card"]["benchmark"] == benchmark("whisperspeech", f"{s2a}+t2s-small", "cuda-float16")


def test_half_precision_uses_float16_on_gpus_without_bfloat16(sandbox, gpu, monkeypatch):
    monkeypatch.setattr(utilities, "has_bfloat16_support", lambda: False)
    assert ss.apply_change("tts.backend", "kyutaipocket")["ok"]
    pocket = tts_backend("kyutaipocket")
    assert pocket["card"]["dtype"] == "float16"
    assert pocket["card"]["benchmark"] == benchmark("kyutaipocket", "", "cuda-float16")
    assert tts_backend("kyutai")["card"]["dtype"] == "float16"
    assert ss.apply_change("tts.backend", "kokoro")["ok"]
    assert ss.apply_change("tts.option", {"backend": "kokoro", "name": "device", "value": "cuda"})["ok"]
    assert "float16" in tts_backend("kokoro")["precision"]["hint"]


def test_kokoro_and_pocket_stay_on_the_cpu_without_a_gpu(sandbox, cpu):
    kokoro = tts_backend("kokoro")
    assert kokoro["device"]["value"] == "cpu"
    assert next(o for o in kokoro["device"]["options"] if o["value"] == "cuda")["disabled"]
    assert kokoro["card"]["benchmark"] == benchmark("kokoro", "", "cpu-float32")
    assert not ss.apply_change("tts.option", {"backend": "kokoro", "name": "device", "value": "cuda"})["ok"]

    assert ss.apply_change("tts.backend", "kyutaipocket")["ok"]
    pocket = tts_backend("kyutaipocket")
    assert (pocket["device"]["value"], pocket["card"]["dtype"]) == ("cpu", "float32")
    assert pocket["card"]["benchmark"] == benchmark("kyutaipocket", "", "cpu-float32")


def test_only_gpu_only_tts_cards_say_so(sandbox, gpu):
    marked = {b["key"] for b in ss.build_state()["tts"]["backends"] if b["card"]["requires_cuda"]}
    assert marked == {"bark", "whisperspeech", "chattts", "chatterbox", "kyutai"}


def test_kyutai_settings_saved_under_the_old_labels_still_load(sandbox, gpu):
    cfg = config(sandbox)
    cfg["kyutai"] = {"model_display_name": "0.75B (EN, ~2GB VRAM)", "hf_repo": "kyutai/tts-0.75b-en-public", "n_q": 16}
    save(sandbox, cfg)
    assert ss.tts_values(config(sandbox))["kyutai"]["model"] == "0.75B (English)"


def test_every_tts_choice_has_a_benchmark():
    variants = {
        "kyutai": [model.split()[0] for model in ss.KYUTAI_MODELS],
        "bark": ss.TTS_SPECS["bark"]["extras"]["size"]["options"],
        "whisperspeech": [f"{s2a}+{t2s}" for s2a in ss.WHISPER_SPEECH_MODELS["s2a"] for t2s in ss.WHISPER_SPEECH_MODELS["t2s"]],
    }
    for backend, info in ss.TTS_BACKENDS.items():
        runs = set()
        for device in info.get("devices", []):
            for precision in info["precisions"]:
                if device == "cpu":
                    runs.add("cpu-float32")
                elif precision == "half":
                    runs.update({f"cuda-{info['half_dtype']}"} if info.get("half_dtype") else {"cuda-bfloat16", "cuda-float16"})
                else:
                    runs.add(f"cuda-{precision}")
        for variant in variants.get(backend, [""]) if runs else []:
            assert runs <= set(ss.TTS_BENCHMARKS[backend][variant]), (backend, variant)


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


def test_vision_cards_describe_each_model_on_a_gpu(sandbox, gpu, monkeypatch):
    cards = ss.build_state()["vision"]["models"]
    assert [c["name"] for c in cards] == list(ss.VISION_MODELS)
    for card in cards:
        info = ss.VISION_MODELS[card["name"]]
        assert card["url"] == f"https://huggingface.co/{info['repo_id']}"
        assert card["size"] == info["size"]
        assert card["precision"] == "bfloat16" and card["quant"] == info["quant"]
        assert card["requires_cuda"] == bool(info["requires_cuda"])
    monkeypatch.setattr(ss.torch.cuda, "get_device_capability", lambda *args: (7, 5))
    assert {c["precision"] for c in ss.build_state()["vision"]["models"]} == {"float16"}


def test_vision_cards_on_a_cpu(sandbox, cpu):
    cards = ss.build_state()["vision"]["models"]
    assert cards and all(not c["requires_cuda"] for c in cards)
    assert {(c["precision"], c["quant"]) for c in cards} == {("float32", "n/a")}
