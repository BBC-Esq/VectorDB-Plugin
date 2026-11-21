import pytest
import yaml

import core.initialize as initialize


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(initialize, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_a_missing_config_is_recreated_from_the_default(root):
    (root / "Assets").mkdir()
    (root / "Assets" / "config.yaml").write_text("appearance:\n  theme: default\n", encoding="utf-8")
    initialize.prepare_config_file()
    assert yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) == {"appearance": {"theme": "default"}}


def test_a_readable_config_is_left_alone(root):
    (root / "config.yaml").write_text("database:\n  contexts: 5\n", encoding="utf-8")
    initialize.prepare_config_file()
    assert (root / "config.yaml").read_text(encoding="utf-8") == "database:\n  contexts: 5\n"


@pytest.mark.parametrize("content", ["<<<<<<< HEAD\nkey: 1\n=======\nkey: 2\n", "- a\n- b\n", "key: [unclosed\n"])
def test_an_unreadable_config_is_reported_and_kept(root, content):
    (root / "config.yaml").write_text(content, encoding="utf-8")
    with pytest.raises(initialize.ConfigFileError, match="could not be read"):
        initialize.prepare_config_file()
    assert (root / "config.yaml").read_text(encoding="utf-8") == content


def test_startup_folders_are_created_in_the_program_folder(root, monkeypatch, tmp_path_factory):
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    initialize.check_for_necessary_folders()
    assert (root / "Vector_DB").is_dir() and (root / "Models" / "vector").is_dir()
