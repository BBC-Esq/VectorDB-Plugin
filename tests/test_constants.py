import pytest

from core.constants import (
    CHAT_MODELS,
    VECTOR_MODELS,
    VISION_MODELS,
    WHISPER_MODELS,
    THINKING_TAGS,
    PIPELINE_PRESETS,
    SUPPORTED_EXTENSIONS,
)


class TestChatModels:

    def test_not_empty(self):
        assert len(CHAT_MODELS) > 0

    def test_required_keys(self):
        required = {'model', 'repo_id', 'cache_dir', 'function', 'precision'}
        for name, model in CHAT_MODELS.items():
            missing = required - set(model.keys())
            assert not missing, f"CHAT_MODELS['{name}'] missing keys: {missing}"

    def test_precision_values(self):
        valid_precisions = {'float16', 'bfloat16', 'float32'}
        for name, model in CHAT_MODELS.items():
            assert model['precision'] in valid_precisions, (
                f"CHAT_MODELS['{name}'] has invalid precision: {model['precision']}"
            )

    def test_repo_id_format(self):
        for name, model in CHAT_MODELS.items():
            assert '/' in model['repo_id'], (
                f"CHAT_MODELS['{name}'] repo_id missing '/': {model['repo_id']}"
            )


class TestVectorModels:

    def test_not_empty(self):
        assert len(VECTOR_MODELS) > 0

    def test_required_keys(self):
        required = {'name', 'dimensions', 'max_sequence', 'size_mb', 'repo_id', 'cache_dir', 'type', 'parameters', 'precision'}
        for group_name, models in VECTOR_MODELS.items():
            for model in models:
                missing = required - set(model.keys())
                assert not missing, (
                    f"VECTOR_MODELS['{group_name}'] model '{model.get('name', '?')}' missing: {missing}"
                )

    def test_dimensions_positive(self):
        for group_name, models in VECTOR_MODELS.items():
            for model in models:
                assert model['dimensions'] > 0, (
                    f"VECTOR_MODELS['{group_name}'] model '{model['name']}' has non-positive dimensions"
                )

    def test_type_is_vector(self):
        for group_name, models in VECTOR_MODELS.items():
            for model in models:
                assert model['type'] == 'vector', (
                    f"VECTOR_MODELS['{group_name}'] model '{model['name']}' type is not 'vector'"
                )


class TestVisionModels:

    def test_not_empty(self):
        assert len(VISION_MODELS) > 0

    def test_required_keys(self):
        required = {'precision', 'repo_id', 'cache_dir', 'requires_cuda'}
        for name, model in VISION_MODELS.items():
            missing = required - set(model.keys())
            assert not missing, f"VISION_MODELS['{name}'] missing keys: {missing}"


class TestWhisperModels:

    def test_not_empty(self):
        assert len(WHISPER_MODELS) > 0

    def test_required_keys(self):
        required = {'name', 'precision', 'repo_id'}
        for name, model in WHISPER_MODELS.items():
            missing = required - set(model.keys())
            assert not missing, f"WHISPER_MODELS['{name}'] missing keys: {missing}"


class TestDocumentLoaders:

    def test_not_empty(self):
        from db.document_processor import LOADER_MAP
        assert len(LOADER_MAP) > 0

    def test_expected_extensions_present(self):
        from db.document_processor import LOADER_MAP
        expected = {'.pdf', '.docx', '.txt', '.csv', '.html'}
        for ext in expected:
            assert ext in LOADER_MAP, f"Missing expected extension: {ext}"


class TestThinkingTags:

    def test_structure(self):
        for key, value in THINKING_TAGS.items():
            assert isinstance(value, tuple), f"THINKING_TAGS['{key}'] is not a tuple"
            assert len(value) == 2, f"THINKING_TAGS['{key}'] doesn't have 2 elements"

    def test_expected_tags(self):
        assert 'think' in THINKING_TAGS
        assert 'thinking' in THINKING_TAGS

    def test_tag_format(self):
        for key, (open_tag, close_tag) in THINKING_TAGS.items():
            assert open_tag.startswith('<')
            assert close_tag.startswith('</')


class TestPipelinePresets:

    def test_not_empty(self):
        assert len(PIPELINE_PRESETS) > 0

    def test_expected_presets(self):
        expected = {"minimal", "low", "normal", "high", "maximum"}
        assert set(PIPELINE_PRESETS.keys()) == expected

    def test_required_keys(self):
        required = {'ingest_threads', 'ingest_processes', 'split_max_parallel_workers',
                     'tokenize_max_parallel_workers', 'split_worker_batch_size'}
        for name, preset in PIPELINE_PRESETS.items():
            missing = required - set(preset.keys())
            assert not missing, f"PIPELINE_PRESETS['{name}'] missing keys: {missing}"

    def test_values_positive(self):
        for name, preset in PIPELINE_PRESETS.items():
            for key, value in preset.items():
                assert isinstance(value, int), f"PIPELINE_PRESETS['{name}']['{key}'] is not int"
                assert value >= 0, f"PIPELINE_PRESETS['{name}']['{key}'] is negative"


class TestSupportedExtensions:

    def test_not_empty(self):
        assert len(SUPPORTED_EXTENSIONS) > 0

    def test_all_start_with_dot(self):
        for ext in SUPPORTED_EXTENSIONS:
            assert ext.startswith('.'), f"Extension '{ext}' doesn't start with '.'"

    def test_expected_extensions(self):
        expected = {'.pdf', '.docx', '.txt', '.csv', '.html'}
        for ext in expected:
            assert ext in SUPPORTED_EXTENSIONS, f"Missing expected extension: {ext}"
