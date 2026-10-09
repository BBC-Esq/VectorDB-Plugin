import pytest
import yaml


class TestConfigIO:

    def _load_config(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def _save_config(self, path, config):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f)

    def test_load_existing_file(self, config_file, sample_config):
        config = self._load_config(config_file)
        assert config['database']['chunk_size'] == sample_config['database']['chunk_size']

    def test_load_missing_file(self, tmp_path):
        config = self._load_config(tmp_path / "nonexistent.yaml")
        assert config == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "config.yaml"
        data = {'openai': {'api_key': 'sk-test123', 'model': 'gpt-4o'}}
        self._save_config(path, data)
        loaded = self._load_config(path)
        assert loaded == data

    def test_empty_config_roundtrip(self, tmp_path):
        path = tmp_path / "config.yaml"
        self._save_config(path, {})
        loaded = self._load_config(path)
        assert loaded == {}


class TestCredentialPaths:

    def test_hf_token_path(self, sample_config):
        assert 'hf_access_token' in sample_config

    def test_openai_key_path(self, sample_config):
        assert 'api_key' in sample_config['openai']

    def test_minimax_key_path(self, sample_config):
        assert 'api_key' in sample_config['minimax']

    def test_hf_credential_get(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        assert config.get('hf_access_token') is None

    def test_openai_credential_get(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        assert config.get('openai', {}).get('api_key') is None

    def test_minimax_credential_get(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        assert config.get('minimax', {}).get('api_key') is None

    def test_update_openai_key(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        config.setdefault('openai', {})['api_key'] = 'sk-newkey'
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f)
        with open(config_file, 'r') as f:
            reloaded = yaml.safe_load(f)
        assert reloaded['openai']['api_key'] == 'sk-newkey'

    def test_update_creates_section_if_missing(self, tmp_path):
        path = tmp_path / "config.yaml"
        with open(path, 'w') as f:
            yaml.safe_dump({}, f)
        with open(path, 'r') as f:
            config = yaml.safe_load(f) or {}
        config.setdefault('minimax', {})['api_key'] = 'mm-key'
        with open(path, 'w') as f:
            yaml.safe_dump(config, f)
        with open(path, 'r') as f:
            reloaded = yaml.safe_load(f)
        assert reloaded['minimax']['api_key'] == 'mm-key'
