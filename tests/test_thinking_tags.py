import re

import pytest

from core.constants import THINKING_TAGS


class TestThinkingTagsStructure:

    def test_think_tag_defined(self):
        assert 'think' in THINKING_TAGS

    def test_thinking_tag_defined(self):
        assert 'thinking' in THINKING_TAGS

    def test_think_tag_values(self):
        assert THINKING_TAGS['think'] == ('<think>', '</think>')

    def test_thinking_tag_values(self):
        assert THINKING_TAGS['thinking'] == ('<thinking>', '</thinking>')


class TestThinkingTagFiltering:

    def _filter_thinking(self, text):
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _filter_thinking_alt(self, text):
        return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()

    def test_removes_think_block(self):
        text = "<think>Internal reasoning here</think>The actual answer."
        result = self._filter_thinking(text)
        assert 'Internal reasoning' not in result
        assert 'The actual answer.' in result

    def test_removes_thinking_block(self):
        text = "<thinking>Some thought process</thinking>Real response."
        result = self._filter_thinking_alt(text)
        assert 'Some thought process' not in result
        assert 'Real response.' in result

    def test_preserves_non_thinking_content(self):
        text = "This is a normal response without thinking tags."
        result = self._filter_thinking(text)
        assert result == text

    def test_multiline_thinking_removed(self):
        text = "<think>\nLine 1\nLine 2\nLine 3\n</think>Answer here."
        result = self._filter_thinking(text)
        assert 'Line 1' not in result
        assert 'Answer here.' in result

    def test_empty_after_filtering(self):
        text = "<think>Only thinking, no answer</think>"
        result = self._filter_thinking(text)
        assert result == ""

    def test_multiple_think_blocks(self):
        text = "<think>First</think>Hello <think>Second</think>World"
        result = self._filter_thinking(text)
        assert 'First' not in result
        assert 'Second' not in result
        assert 'Hello' in result
        assert 'World' in result


class TestLMStudioHiddenThinking:

    def _answer(self, monkeypatch, pieces):
        from types import SimpleNamespace

        import openai

        import chat.lm_studio as lm_studio

        class FakeClient:
            def __init__(self, **kwargs):
                create = lambda **request: [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
                                            for piece in pieces]
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        monkeypatch.setattr(openai, "OpenAI", FakeClient)
        monkeypatch.setattr(lm_studio, "load_chat_config",
                            lambda: {"server": {"connection_str": "http://127.0.0.1:1/v1", "show_thinking": False}})
        return "".join(lm_studio.LMStudioChat().connect_to_local_chatgpt("question"))

    def test_thinking_is_hidden_and_the_answer_kept(self, monkeypatch):
        assert self._answer(monkeypatch, ["<thi", "nk>reasoning</think>", "The answer."]) == "The answer."
        assert self._answer(monkeypatch, ["\n\n<think>abc</th", "ink>\n\nThe answer."]) == "The answer."
        assert self._answer(monkeypatch, ["<thinking>a</thinking><think>b</think>Two blocks."]) == "Two blocks."
        assert self._answer(monkeypatch, ["No thinking ", "at all."]) == "No thinking at all."

    def test_a_tag_mentioned_in_the_answer_is_left_as_text(self, monkeypatch):
        pieces = ["<think>plan</think>", "Reasoning models wrap thoughts in a ", "<think> tag, ", "which is hidden."]
        assert self._answer(monkeypatch, pieces) == "Reasoning models wrap thoughts in a <think> tag, which is hidden."
        assert self._answer(monkeypatch, ["Thoughts go in a <think> tag."]) == "Thoughts go in a <think> tag."

    def test_stopping_while_still_thinking_is_reported(self, monkeypatch):
        from chat.lm_studio import STOPPED_WHILE_THINKING

        with pytest.raises(RuntimeError, match="still thinking") as error:
            self._answer(monkeypatch, ["<think>", "I should look at the lease first"])
        assert str(error.value) == STOPPED_WHILE_THINKING
