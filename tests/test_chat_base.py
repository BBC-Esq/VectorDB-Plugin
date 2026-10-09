import pytest
import torch

from chat.base import (
    build_augmented_query,
    get_max_length,
    get_max_new_tokens,
    get_generation_settings,
    make_bnb_settings,
)
from core.constants import rag_string, CHAT_MODELS


class TestBuildAugmentedQuery:

    def test_includes_all_contexts(self):
        contexts = ["Context A", "Context B", "Context C"]
        result = build_augmented_query(contexts, "What is X?")
        for ctx in contexts:
            assert ctx in result

    def test_includes_query(self):
        result = build_augmented_query(["ctx"], "What is X?")
        assert "What is X?" in result

    def test_includes_rag_string(self):
        result = build_augmented_query(["ctx"], "query")
        assert rag_string in result

    def test_separators_present(self):
        result = build_augmented_query(["ctx1", "ctx2"], "query")
        assert "---" in result

    def test_single_context(self):
        result = build_augmented_query(["only context"], "my question")
        assert "only context" in result
        assert "my question" in result

    def test_query_at_end(self):
        result = build_augmented_query(["ctx"], "final question")
        last_separator_idx = result.rfind("-----")
        query_idx = result.find("final question")
        assert query_idx > last_separator_idx


class TestMaxLength:

    def test_known_model(self):
        for name, info in CHAT_MODELS.items():
            if 'max_tokens' in info:
                result = get_max_length(name)
                assert result == info['max_tokens']
                break

    def test_unknown_model_default(self):
        result = get_max_length("nonexistent_model_xyz")
        assert result == 8192


class TestMaxNewTokens:

    def test_known_model(self):
        for name, info in CHAT_MODELS.items():
            if 'max_new_tokens' in info:
                result = get_max_new_tokens(name)
                assert result == info['max_new_tokens']
                break

    def test_unknown_model_default(self):
        result = get_max_new_tokens("nonexistent_model_xyz")
        assert result == 1024


class TestGenerationSettings:

    def test_keys(self):
        settings = get_generation_settings(8192, 1024)
        expected_keys = {'max_length', 'max_new_tokens', 'do_sample', 'num_beams', 'use_cache', 'temperature', 'top_p', 'top_k'}
        assert set(settings.keys()) == expected_keys

    def test_values(self):
        settings = get_generation_settings(4096, 512)
        assert settings['max_length'] == 4096
        assert settings['max_new_tokens'] == 512
        assert settings['do_sample'] is False
        assert settings['num_beams'] == 1
        assert settings['use_cache'] is True
        assert settings['temperature'] is None


class TestMakeBnbSettings:

    def test_structure(self):
        settings = make_bnb_settings(torch.float16)
        assert 'tokenizer_settings' in settings
        assert 'model_settings' in settings

    def test_tokenizer_settings_empty(self):
        settings = make_bnb_settings(torch.bfloat16)
        assert settings['tokenizer_settings'] == {}

    def test_model_dtype(self):
        settings = make_bnb_settings(torch.float16)
        assert settings['model_settings']['dtype'] == torch.float16

    def test_quantization_config(self):
        settings = make_bnb_settings(torch.float16)
        qc = settings['model_settings']['quantization_config']
        assert qc.load_in_4bit is True
        assert qc.bnb_4bit_quant_type == "nf4"
        assert qc.bnb_4bit_use_double_quant is True

    def test_no_deprecated_keys(self):
        settings = make_bnb_settings(torch.float16)
        assert 'torch_dtype' not in settings['model_settings']
        assert 'low_cpu_mem_usage' not in settings['model_settings']
