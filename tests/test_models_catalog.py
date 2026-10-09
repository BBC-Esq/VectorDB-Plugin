import pytest
import torch

from core.constants import THEMES, VECTOR_MODELS
from core.utilities import get_embedding_dtype_and_batch
from gui.tabs_models import catalog
from gui.web_common.theme import theme_payload


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(catalog, "cuda_usable", lambda: False)
    monkeypatch.setattr("core.utilities.cuda_usable", lambda: False)
    catalog.detect_hardware.cache_clear()
    yield
    catalog.detect_hardware.cache_clear()


def all_models():
    return [model for models in VECTOR_MODELS.values() for model in models]


def test_parse_parameters():
    assert catalog.parse_parameters("33.4m") == pytest.approx(33.4)
    assert catalog.parse_parameters("1720m") == pytest.approx(1720)
    assert catalog.parse_parameters("7.57b") == pytest.approx(7570)
    assert catalog.parse_parameters("unknown") == 0.0


def test_every_model_has_a_known_license_and_parameters():
    for model in all_models():
        assert model["license"].lower() in catalog.LICENSES, model["name"]
        assert catalog.parse_parameters(model["parameters"]) > 0, model["name"]


def test_find_model():
    assert catalog.find_model("BAAI/bge-small-en-v1.5")["name"] == "bge-small-en-v1.5"
    assert catalog.find_model("nobody/nothing") is None


def test_cpu_payload_hides_gpu_only_models_and_runs_float32(cpu_only, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "read_precision_settings", lambda: ("cuda", True))
    monkeypatch.chdir(tmp_path)
    payload = catalog.build_payload()
    expected = [m["repo_id"] for m in all_models() if not m.get("requires_cuda")]
    assert [m["id"] for m in payload["models"]] == expected
    assert payload["hardware"]["cpu_only"] is True
    assert payload["hardware"]["half"] is False
    assert payload["cpu_note"]
    assert all(m["precision"]["current"] == "float32" for m in payload["models"])
    assert not any(m["downloaded"] for m in payload["models"])


def test_downloaded_status_ignores_incomplete_folders(cpu_only, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "read_precision_settings", lambda: ("cpu", False))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Models" / "vector" / "BAAI--bge-small-en-v1.5").mkdir(parents=True)
    partial = tmp_path / "Models" / "vector" / "intfloat--e5-small-v2"
    partial.mkdir(parents=True)
    (partial / ".download_incomplete").touch()
    status = {m["id"]: m["downloaded"] for m in catalog.build_payload()["models"]}
    assert status["BAAI/bge-small-en-v1.5"] is True
    assert status["intfloat/e5-small-v2"] is False
    assert sum(status.values()) == 1


def test_benchmark_summary_matches_the_scores(cpu_only, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "read_precision_settings", lambda: ("cpu", False))
    monkeypatch.chdir(tmp_path)
    payload = catalog.build_payload()
    by_id = {m["id"]: m for m in payload["models"]}
    for bench in payload["benchmarks"]:
        scores = [by_id[i]["scores"][bench["key"]] for i in bench["ranking"]]
        assert scores == sorted(scores, reverse=True)
        assert bench["count"] == len(scores)
        if scores:
            assert bench["max"] == scores[0] and bench["min"] == scores[-1]
        unscored = [m for m in payload["models"] if m["scores"][bench["key"]] is None]
        assert len(unscored) + len(scores) == len(payload["models"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gpu_precision_matches_the_embedding_code(monkeypatch, tmp_path):
    catalog.detect_hardware.cache_clear()
    monkeypatch.chdir(tmp_path)
    for half in (False, True):
        monkeypatch.setattr(catalog, "read_precision_settings", lambda half=half: ("cuda", half))
        payload = catalog.build_payload()
        assert payload["hardware"]["half"] is half
        for m in payload["models"]:
            info = catalog.find_model(m["id"])
            dtype, _ = get_embedding_dtype_and_batch(
                "cuda", half, info["precision"], catalog.model_directory_name(info), True
            )
            assert m["precision"]["current"] == str(dtype).split(".")[-1], m["name"]


def test_downloading_is_passed_through(cpu_only, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "read_precision_settings", lambda: ("cpu", False))
    monkeypatch.chdir(tmp_path)
    assert catalog.build_payload("BAAI/bge-small-en-v1.5")["downloading"] == "BAAI/bge-small-en-v1.5"
    assert catalog.build_payload()["downloading"] is None


def test_theme_payload_covers_every_theme():
    for name, colors in THEMES.items():
        theme = theme_payload(name)
        assert set(theme["colors"]) == set(colors)
        assert theme["scheme"] in ("light", "dark")
    assert theme_payload("colorblind")["scheme"] == "light"
    assert theme_payload("steel_ocean")["scheme"] == "dark"
    tron = theme_payload("tron")
    assert tron["on_control_hover"] != tron["colors"]["text_primary"]
    assert theme_payload("no-such-theme")["colors"] == theme_payload("default")["colors"]
