import zipfile
from pathlib import Path


def _project_with_manual(tmp_path, monkeypatch):
    import tools.replace_sourcecode as replace_sourcecode

    monkeypatch.setattr(replace_sourcecode, "PROJECT_ROOT", tmp_path)
    (tmp_path / "Assets").mkdir()
    manual = tmp_path / "Vector_DB" / "user_manual"
    manual.mkdir(parents=True)
    (manual / "metadata.db").write_text("installed manual", encoding="utf-8")
    return replace_sourcecode, manual


def test_an_unreadable_manual_archive_leaves_the_installed_manual_alone(tmp_path, monkeypatch):
    replace_sourcecode, manual = _project_with_manual(tmp_path, monkeypatch)
    (tmp_path / "Assets" / "user_manual_db.zip").write_text("not a zip file", encoding="utf-8")
    replace_sourcecode.setup_vector_db()
    assert (manual / "metadata.db").read_text(encoding="utf-8") == "installed manual"


def test_a_good_manual_archive_replaces_the_installed_manual(tmp_path, monkeypatch):
    replace_sourcecode, manual = _project_with_manual(tmp_path, monkeypatch)
    (manual / "stale.txt").write_text("old", encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "Assets" / "user_manual_db.zip", "w") as archive:
        archive.writestr("user_manual/metadata.db", "new manual")
    replace_sourcecode.setup_vector_db()
    for folder in ("Vector_DB", "Vector_DB_Backup"):
        assert (tmp_path / folder / "user_manual" / "metadata.db").read_text(encoding="utf-8") == "new manual"
    assert not (manual / "stale.txt").exists()
