import pytest
import requests
import yaml

from gui.dialogs import ai_backends_data as data

LONG_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789WXYZ"


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "PROJECT_ROOT", tmp_path)
    return tmp_path


def write_config(root, cfg):
    (root / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def read_config(root):
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, status, payload=None, bad_json=False):
        self.status_code = status
        self._payload = payload
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


def fake_get(monkeypatch, response=None, error=None, calls=None):
    def get(url, **kwargs):
        if calls is not None:
            calls.append((url, kwargs))
        if error:
            raise error
        return response
    monkeypatch.setattr(data.requests, "get", get)


def test_normalize_replaces_models_and_choices_that_are_no_longer_offered(root):
    write_config(root, {"openai": {"api_key": "k", "model": "gpt-4o", "verbosity": "loud", "reasoning_effort": "high"}, "other": 1})
    assert data.normalize_config() is True
    cfg = read_config(root)
    assert cfg["openai"] == {"api_key": "k", "model": data.DEFAULT_OPENAI_MODEL, "verbosity": data.DEFAULT_VERBOSITY, "reasoning_effort": "high"}
    assert cfg["other"] == 1


def test_normalize_leaves_valid_or_missing_values_alone(root):
    write_config(root, {"openai": {"model": "gpt-5.4", "verbosity": None}})
    before = (root / "config.yaml").read_bytes()
    assert data.normalize_config() is False
    assert (root / "config.yaml").read_bytes() == before
    write_config(root, {"server": {}})
    assert data.normalize_config() is False


def test_normalize_never_touches_an_unreadable_or_missing_config(root):
    assert data.normalize_config() is False
    (root / "config.yaml").write_text("openai: [broken\n", encoding="utf-8")
    assert data.normalize_config() is False
    assert (root / "config.yaml").read_text(encoding="utf-8") == "openai: [broken\n"


def test_api_keys_are_trimmed_saved_and_cleared(root):
    write_config(root, {"database": {"chunk_size": 700}})
    assert data.apply_change("openai.api_key", f"  {LONG_KEY}  ") == {"ok": True, "changed": True}
    assert read_config(root)["openai"]["api_key"] == LONG_KEY
    assert data.apply_change("openai.api_key", LONG_KEY) == {"ok": True, "changed": False}
    assert data.apply_change("openai.api_key", "") == {"ok": True, "changed": True}
    assert read_config(root)["openai"]["api_key"] is None
    assert read_config(root)["database"] == {"chunk_size": 700}
    assert data.apply_change("minimax.api_key", "mm-key-123") == {"ok": True, "changed": True}
    assert data.saved_key("minimax") == "mm-key-123"
    assert data.saved_key("chatgpt") == ""


def test_api_keys_with_spaces_are_refused(root):
    write_config(root, {})
    result = data.apply_change("openai.api_key", "sk-proj two parts")
    assert result == {"ok": False, "error": "An API key can't contain spaces."}
    assert read_config(root) == {}


def test_choices_must_be_offered(root):
    write_config(root, {})
    assert data.apply_change("openai.model", "gpt-5.5")["changed"] is True
    assert data.apply_change("openai.verbosity", "high")["changed"] is True
    assert data.apply_change("openai.reasoning_effort", "xhigh")["changed"] is True
    assert read_config(root)["openai"] == {"model": "gpt-5.5", "verbosity": "high", "reasoning_effort": "xhigh"}
    assert data.apply_change("openai.model", "gpt-4o")["ok"] is False
    assert data.apply_change("openai.verbosity", "loud")["ok"] is False
    assert data.apply_change("openai.reasoning_effort", "max")["ok"] is False
    assert data.apply_change("openai.unknown", 1) == {"ok": False, "error": "Unknown setting: openai.unknown"}


def test_port_replaces_only_the_port_of_the_saved_address(root):
    write_config(root, {"server": {"connection_str": "http://192.168.1.5:1234/v1", "show_thinking": True}})
    assert data.apply_change("server.port", "08080") == {"ok": True, "changed": True}
    assert read_config(root)["server"] == {"connection_str": "http://192.168.1.5:8080/v1", "show_thinking": True}
    assert data.apply_change("server.port", "8080")["changed"] is False


def test_port_creates_the_standard_address_when_none_is_saved(root):
    write_config(root, {"server": {"show_thinking": False}})
    assert data.apply_change("server.port", "1234")["ok"] is True
    assert read_config(root)["server"]["connection_str"] == "http://localhost:1234/v1"


@pytest.mark.parametrize("value", ["0", "65536", "abc", "", "12.5", "123456"])
def test_impossible_ports_are_refused(root, value):
    write_config(root, {"server": {"connection_str": "http://localhost:1234/v1"}})
    assert data.apply_change("server.port", value) == {"ok": False, "error": "Port must be a number between 1 and 65535."}
    assert read_config(root)["server"]["connection_str"] == "http://localhost:1234/v1"


def test_an_address_without_a_port_is_left_alone(root):
    write_config(root, {"server": {"connection_str": "http://lmstudio-box/v1"}})
    result = data.apply_change("server.port", "1500")
    assert result["ok"] is False and "has no port to change" in result["error"]
    assert read_config(root)["server"]["connection_str"] == "http://lmstudio-box/v1"


def test_show_thinking_and_unreadable_configs(root):
    write_config(root, {})
    assert data.apply_change("server.show_thinking", True)["changed"] is True
    assert data.apply_change("server.show_thinking", True)["changed"] is False
    assert read_config(root)["server"]["show_thinking"] is True
    (root / "config.yaml").write_text("server: [broken\n", encoding="utf-8")
    result = data.apply_change("server.show_thinking", False)
    assert result["ok"] is False and "config.yaml could not be read" in result["error"]
    assert (root / "config.yaml").read_text(encoding="utf-8") == "server: [broken\n"


def test_build_state_describes_each_backend(root):
    write_config(root, {
        "openai": {"api_key": LONG_KEY, "model": "gpt-5.5"},
        "server": {"connection_str": "http://localhost:4321/v1", "show_thinking": True},
        "minimax": {"api_key": "short"},
    })
    state = data.build_state("lmstudio", {"kobold": {"status": "ok", "message": "fine"}})
    assert state["selected"] == "lmstudio" and state["error"] is None
    chatgpt = state["chatgpt"]
    assert chatgpt["key_set"] and chatgpt["key_hint"] == "WXYZ" and chatgpt["model"] == "gpt-5.5"
    assert chatgpt["pricing"] == {"input": 5.0, "cached": 0.5, "output": 30.0}
    assert [m["value"] for m in chatgpt["models"]] == data.AVAILABLE_OPENAI_MODELS
    assert chatgpt["models"][0]["note"] == "$5.00 in · $30.00 out"
    assert chatgpt["show_verbosity"] and chatgpt["show_reasoning"]
    assert state["lmstudio"]["port"] == "4321" and not state["lmstudio"]["missing"] and not state["lmstudio"]["malformed"]
    assert state["lmstudio"]["show_thinking"] is True
    assert state["minimax"]["key_set"] and state["minimax"]["key_hint"] == ""
    assert state["minimax"]["models"] == data.MINIMAX_MODELS
    assert state["kobold"] == {"address": data.KOBOLD_URL, "test": {"status": "ok", "message": "fine"}}
    assert data.build_state("nonsense")["selected"] == "chatgpt"


def test_build_state_flags_missing_and_malformed_addresses_and_unreadable_configs(root):
    write_config(root, {"server": {}})
    assert data.build_state()["lmstudio"]["missing"] is True
    write_config(root, {"server": {"connection_str": "http://box/v1"}})
    lm = data.build_state()["lmstudio"]
    assert lm["malformed"] is True and lm["port"] == ""
    (root / "config.yaml").write_text("x: [", encoding="utf-8")
    state = data.build_state()
    assert "config.yaml could not be read" in state["error"] and state["chatgpt"]["key_set"] is False


def test_chatgpt_check_reports_each_answer(root, monkeypatch):
    cfg = {"openai": {"api_key": LONG_KEY, "model": "gpt-5.4-mini"}}
    assert data.check_chatgpt({})["message"] == "Enter an API key first."
    calls = []
    fake_get(monkeypatch, FakeResponse(200, {"data": [{"id": "gpt-5.4-mini"}]}), calls=calls)
    assert data.check_chatgpt(cfg) == {"status": "ok", "message": "The key works and can use gpt-5.4 mini."}
    assert calls[0][1]["headers"] == {"Authorization": f"Bearer {LONG_KEY}"} and calls[0][1]["timeout"] == 15
    fake_get(monkeypatch, FakeResponse(200, {"data": [{"id": "gpt-5.4"}]}))
    assert data.check_chatgpt(cfg)["status"] == "warn"
    fake_get(monkeypatch, FakeResponse(401, {}))
    assert "did not accept this API key" in data.check_chatgpt(cfg)["message"]
    fake_get(monkeypatch, FakeResponse(429, {}))
    assert data.check_chatgpt(cfg)["status"] == "warn"
    fake_get(monkeypatch, FakeResponse(503, {}))
    assert data.check_chatgpt(cfg)["message"] == "OpenAI answered with an error (HTTP 503). Try again later."
    fake_get(monkeypatch, error=requests.ConnectionError("down"))
    assert "Could not reach OpenAI" in data.check_chatgpt(cfg)["message"]


def test_lmstudio_check_reports_each_answer(root, monkeypatch):
    cfg = {"server": {"connection_str": "http://localhost:1234/v1/"}}
    assert data.check_lmstudio({})["message"] == "Enter LM Studio's port first."
    calls = []
    fake_get(monkeypatch, FakeResponse(200, {"data": [{"id": n} for n in ("a", "b", "c", "d", "e")]}), calls=calls)
    assert data.check_lmstudio(cfg) == {"status": "ok", "message": "LM Studio is running with a, b, c and 2 more."}
    assert calls[0][0] == "http://localhost:1234/v1/models"
    fake_get(monkeypatch, FakeResponse(200, {"data": []}))
    assert data.check_lmstudio(cfg)["status"] == "warn"
    fake_get(monkeypatch, FakeResponse(404, {}))
    assert "doesn't look like LM Studio (HTTP 404)" in data.check_lmstudio(cfg)["message"]
    fake_get(monkeypatch, FakeResponse(200, bad_json=True))
    assert data.check_lmstudio(cfg)["status"] == "error"
    fake_get(monkeypatch, error=requests.ConnectionError("refused"))
    assert "isn't answering at http://localhost:1234/v1/" in data.check_lmstudio(cfg)["message"]


def test_kobold_check_reports_each_answer(root, monkeypatch):
    calls = []
    fake_get(monkeypatch, FakeResponse(200, {"result": "koboldcpp/Mistral-7B"}), calls=calls)
    assert data.check_kobold({}) == {"status": "ok", "message": "KoboldCpp is running Mistral-7B."}
    assert calls[0][0] == f"{data.KOBOLD_URL}/api/v1/model"
    fake_get(monkeypatch, FakeResponse(200, {"result": ""}))
    assert data.check_kobold({})["message"] == "KoboldCpp is running."
    fake_get(monkeypatch, FakeResponse(500, {}))
    assert data.check_kobold({})["status"] == "error"
    fake_get(monkeypatch, error=requests.Timeout("slow"))
    assert "isn't answering" in data.check_kobold({})["message"]


def test_kobold_address_matches_the_backend():
    from chat.kobold import KOBOLD_URL

    assert data.KOBOLD_URL == KOBOLD_URL == "http://localhost:5001"
