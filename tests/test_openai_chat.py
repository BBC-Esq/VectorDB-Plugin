import sys
from types import SimpleNamespace

import pytest

from chat.openai import ChatGPTChat
from core import chatgpt_settings as settings


def test_models_and_their_reasoning_efforts():
    assert settings.AVAILABLE_OPENAI_MODELS == ["gpt-6-astra", "gpt-6.1-sol", "gpt-6-luna"]
    assert settings.DEFAULT_OPENAI_MODEL == "gpt-6-luna"
    assert settings.reasoning_effort_options("gpt-6-astra") == ["low", "medium", "high", "xhigh", "max"]
    assert settings.reasoning_effort_options("gpt-6.1-sol") == ["low", "medium", "high", "xhigh", "max"]
    assert settings.reasoning_effort_options("gpt-6-luna") == ["none", "low", "medium", "high", "xhigh", "max"]
    assert settings.reasoning_effort_options("gpt-5.4") == ["none", "low", "medium", "high", "xhigh"]
    assert settings.reasoning_effort_options("gpt-5.4-chat-latest") == []
    for model in settings.AVAILABLE_OPENAI_MODELS:
        assert settings.supports_verbosity(model) and settings.supports_reasoning_effort(model)
        assert settings.get_model_pricing(model) != (0.0, 0.0, 0.0)


@pytest.mark.parametrize("model, asked, used", [
    ("gpt-6-astra", "none", "low"),
    ("gpt-6.1-sol", "max", "max"),
    ("gpt-5.4", "max", "xhigh"),
    ("gpt-6-luna", "extreme", "medium"),
    ("gpt-6-luna", "none", "none"),
])
def test_an_effort_a_model_lacks_is_replaced_by_the_nearest_one(model, asked, used):
    assert settings.clamp_reasoning_effort(model, asked) == used


def test_models_no_longer_offered_fall_back_to_the_default():
    assert settings.migrate_legacy_model("gpt-6.1-sol") == "gpt-6.1-sol"
    for old in ("gpt-5.4-mini", "gpt-4o", None, ""):
        assert settings.migrate_legacy_model(old) == "gpt-6-luna"


@pytest.fixture
def openai_calls(monkeypatch):
    calls = []
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_text.delta", delta="Hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
        SimpleNamespace(type="response.completed"),
    ]

    class FakeOpenAI:
        def __init__(self, api_key):
            self.responses = SimpleNamespace(create=lambda **kwargs: calls.append((api_key, kwargs)) or iter(events))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    return calls


def ask(openai_section):
    chat = ChatGPTChat.__new__(ChatGPTChat)
    chat.config = {"openai": {"api_key": "sk-test", **openai_section}}
    return "".join(chat.connect_to_chatgpt("question with contexts"))


@pytest.mark.parametrize("saved, model, effort, verbosity", [
    ({"model": "gpt-6-luna", "reasoning_effort": "none", "verbosity": "high"}, "gpt-6-luna", "none", "high"),
    ({"model": "gpt-6-astra", "reasoning_effort": "none"}, "gpt-6-astra", "low", "low"),
    ({"model": "gpt-6.1-sol", "reasoning_effort": "max", "verbosity": "loud"}, "gpt-6.1-sol", "max", "low"),
    ({"model": "gpt-5.4-mini"}, "gpt-6-luna", "medium", "low"),
])
def test_the_request_fits_the_chosen_model(openai_calls, saved, model, effort, verbosity):
    assert ask(saved) == "Hello"
    api_key, request = openai_calls[0]
    assert api_key == "sk-test"
    assert request["model"] == model and request["stream"] is True
    assert request["reasoning"] == {"effort": effort}
    assert request["text"] == {"verbosity": verbosity}
    assert [message["role"] for message in request["input"]] == ["system", "user"]
    assert request["input"][1]["content"] == "question with contexts"


def test_errors_from_openai_are_raised(monkeypatch):
    class FailingOpenAI:
        def __init__(self, api_key):
            failure = SimpleNamespace(type="response.failed", response=SimpleNamespace(error=SimpleNamespace(message="model overloaded")))
            self.responses = SimpleNamespace(create=lambda **kwargs: iter([failure]))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FailingOpenAI))
    with pytest.raises(RuntimeError, match="model overloaded"):
        ask({"model": "gpt-6-luna"})


def test_a_missing_api_key_is_reported_before_contacting_openai(openai_calls):
    chat = ChatGPTChat.__new__(ChatGPTChat)
    chat.config = {"openai": {"model": "gpt-6-luna"}}
    with pytest.raises(ValueError, match="API key not found"):
        "".join(chat.connect_to_chatgpt("question"))
    assert openai_calls == []
