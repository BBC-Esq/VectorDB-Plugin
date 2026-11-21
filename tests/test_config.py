from core.config import AppConfig


def test_environment_variables_are_not_read_as_settings(tmp_path, monkeypatch):
    for name, value in (("DATABASE", "prod"), ("SERVER", "myhost"), ("OPENAI", "x"), ("HF_ACCESS_TOKEN", "from-env")):
        monkeypatch.setenv(name, value)
    (tmp_path / "config.yaml").write_text("database:\n  contexts: 7\n", encoding="utf-8")
    config = AppConfig.load(tmp_path / "config.yaml")
    assert config.database.contexts == 7
    assert config.hf_access_token is None
