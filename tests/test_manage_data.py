import json
import os
import sqlite3

import pytest
import yaml

from db.document_processor import Document
from db.sqlite_operations import create_metadata_db
from gui.tabs_databases import manage_data


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(manage_data, "PROJECT_ROOT", tmp_path)
    (tmp_path / "Vector_DB").mkdir()
    (tmp_path / "Vector_DB_Backup").mkdir()
    return tmp_path


def write_config(root, data):
    (root / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def read_config(root):
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def make_db(root, name, files, chunks_per_file=3, backup=False):
    folder = root / "Vector_DB" / name
    docs = []
    mappings = []
    for i, path in enumerate(files):
        digest = f"{name}-{i}"
        docs.append(Document(page_content="text", metadata={"file_name": os.path.basename(path), "file_path": path, "hash": digest}))
        mappings.extend((f"{digest}-{j}", digest) for j in range(chunks_per_file))
    create_metadata_db(folder, docs, mappings)
    (folder / "index_metadata.json").write_text(json.dumps({"dimensions": 384, "num_vectors": len(mappings)}), encoding="utf-8")
    (folder / "vectors").mkdir()
    (folder / "vectors" / "data.bin").write_bytes(b"x" * 1000)
    if backup:
        import shutil
        shutil.copytree(folder, root / "Vector_DB_Backup" / name)
    return folder


def entry(model="M:/Models/vector/BAAI--bge-small-en-v1.5"):
    return {"model": model, "chunk_size": 900, "chunk_overlap": 300}


def test_kinds():
    assert manage_data.kind_of("C:/a/Report.PDF") == "pdf"
    assert manage_data.kind_of("call.wav") == "audio"
    assert manage_data.kind_of("notes.rtf") == "text"
    assert manage_data.kind_of("archive.zip") == "other"


def test_list_hides_jeeves_and_marks_states(root):
    make_db(root, "ready_db", ["C:/docs/a.txt"], backup=True)
    make_db(root, "user_manual", ["C:/docs/m.txt"])
    (root / "Vector_DB" / "leftover").mkdir()
    (root / "Vector_DB" / ".hidden_temp").mkdir()
    write_config(root, {"created_databases": {"ready_db": entry(), "missing_db": entry(), "user_manual": entry(), 2024: entry()}})
    rows = {r["name"]: r for r in manage_data.list_databases(manage_data.read_config())}
    assert sorted(rows) == ["2024", "leftover", "missing_db", "ready_db"]
    assert (rows["ready_db"]["registered"], rows["ready_db"]["folder"], rows["ready_db"]["backup"]) == (True, True, True)
    assert (rows["missing_db"]["registered"], rows["missing_db"]["folder"]) == (True, False)
    assert (rows["leftover"]["registered"], rows["leftover"]["folder"]) == (False, True)
    assert rows["ready_db"]["chunk_size"] == 900


def test_read_config_errors(root):
    with pytest.raises(manage_data.ConfigError):
        manage_data.read_config()
    (root / "config.yaml").write_text("created_databases: [broken\n", encoding="utf-8")
    with pytest.raises(manage_data.ConfigError):
        manage_data.read_config()
    (root / "config.yaml").write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(manage_data.ConfigError):
        manage_data.read_config()


def test_database_info_and_incomplete(root):
    make_db(root, "ready_db", ["C:/docs/a.txt", "C:/docs/b.pdf"], chunks_per_file=4)
    info = manage_data.database_info("ready_db")
    assert (info["files"], info["chunks"], info["dimensions"], info["complete"]) == (2, 8, 384, True)
    assert info["size"] >= 1000 and info["created"]
    (root / "Vector_DB" / "half").mkdir()
    half = manage_data.database_info("half")
    assert half["complete"] is False and "metadata.db" in half["problem"]
    (root / "Vector_DB" / "ready_db" / "index_metadata.json").unlink()
    no_index = manage_data.database_info("ready_db")
    assert no_index["complete"] is False and no_index["chunks"] == 8 and "index_metadata.json" in no_index["problem"]


def test_file_rows_compact_and_missing(root, tmp_path_factory):
    source = tmp_path_factory.mktemp("originals")
    present = source / "present.pdf"
    present.write_text("x", encoding="utf-8")
    gone = str(source / "gone.txt")
    make_db(root, "files_db", [str(present), gone], chunks_per_file=5)
    rows = manage_data.file_rows("files_db")
    assert [r[0] for r in rows] == ["gone.txt", "present.pdf"] and [r[2] for r in rows] == [5, 5]
    compact = manage_data.compact_rows(rows)
    assert compact["dirs"] == [str(source)]
    assert compact["rows"] == [["gone.txt", "text", 5, 0], ["present.pdf", "pdf", 5, 0]]
    assert manage_data.missing_rows(rows, lambda: False) == [0]
    assert manage_data.missing_rows(rows, lambda: True) is None


def test_compact_rows_keeps_odd_paths():
    rows = [("call.wav", "D:/audio/call.wav", 2), ("label.txt", "D:/other/real_name.txt", 1), ("nothing", "", 0)]
    compact = manage_data.compact_rows(rows)
    assert compact["rows"][0] == ["call.wav", "audio", 2, 0]
    assert compact["rows"][1] == ["label.txt", "text", 1, -1, "D:/other/real_name.txt"]
    assert compact["rows"][2] == ["nothing", "other", 0, -1, ""]


def test_file_rows_shared_hash_split(root):
    folder = root / "Vector_DB" / "dupes"
    docs = [Document(page_content="t", metadata={"file_name": n, "file_path": f"C:/d/{n}", "hash": "same"}) for n in ("a.txt", "b.txt")]
    create_metadata_db(folder, docs, [(f"id{i}", "same") for i in range(6)])
    assert [r[2] for r in manage_data.file_rows("dupes")] == [3, 3]


def test_readonly_connection_never_writes(root):
    folder = make_db(root, "ro_db", ["C:/docs/a.txt"])
    before = (folder / "metadata.db").stat().st_mtime_ns
    with pytest.raises(sqlite3.OperationalError):
        manage_data.connect_readonly(folder / "metadata.db").execute("DELETE FROM document_metadata")
    manage_data.file_rows("ro_db")
    assert (folder / "metadata.db").stat().st_mtime_ns == before


def test_delete_removes_folders_and_entry(root):
    make_db(root, "doomed", ["C:/docs/a.txt"], backup=True)
    make_db(root, "keeper", ["C:/docs/b.txt"])
    write_config(root, {"created_databases": {"doomed": entry(), "keeper": entry()},
                        "database": {"database_to_search": "keeper"}, "other": {"api_key": "secret"}})
    assert manage_data.delete_database("doomed") == {"failed": []}
    cfg = read_config(root)
    assert list(cfg["created_databases"]) == ["keeper"] and cfg["database"]["database_to_search"] == "keeper"
    assert cfg["other"]["api_key"] == "secret"
    assert not (root / "Vector_DB" / "doomed").exists() and not (root / "Vector_DB_Backup" / "doomed").exists()
    assert (root / "Vector_DB" / "keeper").exists()


def test_delete_clears_matching_search_target_and_int_keys(root):
    make_db(root, "2024", ["C:/docs/a.txt"])
    write_config(root, {"created_databases": {2024: entry()}, "database": {"database_to_search": "2024"}})
    manage_data.delete_database("2024")
    cfg = read_config(root)
    assert cfg["created_databases"] == {} and cfg["database"]["database_to_search"] == ""


def test_delete_leftover_leaves_config_alone(root):
    (root / "Vector_DB" / "leftover" / "vectors").mkdir(parents=True)
    write_config(root, {"created_databases": {}})
    before = (root / "config.yaml").stat().st_mtime_ns
    assert manage_data.delete_database("leftover") == {"failed": []}
    assert not (root / "Vector_DB" / "leftover").exists()
    assert (root / "config.yaml").stat().st_mtime_ns == before


def test_delete_refuses_on_unreadable_config(root):
    make_db(root, "safe", ["C:/docs/a.txt"])
    (root / "config.yaml").write_text("created_databases: [broken\n", encoding="utf-8")
    with pytest.raises(manage_data.ConfigError):
        manage_data.delete_database("safe")
    assert (root / "Vector_DB" / "safe" / "metadata.db").exists()


def test_delete_reports_locked_files(root, monkeypatch):
    folder = make_db(root, "locked", ["C:/docs/a.txt"])
    write_config(root, {"created_databases": {"locked": entry()}})
    monkeypatch.setattr(manage_data.time, "sleep", lambda s: None)
    with open(folder / "index_metadata.json", "rb"):
        result = manage_data.delete_database("locked")
    assert len(result["failed"]) == 1 and "could not be fully deleted" in result["failed"][0]
    assert "locked" not in read_config(root)["created_databases"]
    assert manage_data.delete_database("locked") == {"failed": []}
    assert not folder.exists()


def test_remove_tree_refuses_links(root, tmp_path_factory):
    target = tmp_path_factory.mktemp("real_data")
    (target / "keep.txt").write_text("x", encoding="utf-8")
    link = root / "Vector_DB" / "linked"
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available")
    problem = manage_data.remove_tree(link)
    assert "is a link" in problem and (target / "keep.txt").exists() and os.path.lexists(link)
    os.rmdir(link) if os.path.isdir(link) and not os.path.islink(link) else os.unlink(link)


def test_backup_and_restore(root):
    make_db(root, "precious", ["C:/docs/a.txt"])
    manage_data.backup_database("precious")
    assert (root / "Vector_DB_Backup" / "precious" / "metadata.db").exists()
    assert not (root / ".backup_precious").exists()
    with pytest.raises(FileExistsError):
        manage_data.restore_database("precious")
    import shutil
    shutil.rmtree(root / "Vector_DB" / "precious")
    manage_data.restore_database("precious")
    assert (root / "Vector_DB" / "precious" / "metadata.db").exists() and not (root / ".restore_precious").exists()
    (root / "Vector_DB" / "half").mkdir()
    with pytest.raises(FileNotFoundError):
        manage_data.backup_database("half")


def test_backup_keeps_old_copy_when_copy_fails(root, monkeypatch):
    make_db(root, "precious", ["C:/docs/a.txt"], backup=True)
    old = (root / "Vector_DB_Backup" / "precious" / "metadata.db").read_bytes()

    def broken(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(manage_data.shutil, "copytree", broken)
    with pytest.raises(OSError):
        manage_data.backup_database("precious")
    assert (root / "Vector_DB_Backup" / "precious" / "metadata.db").read_bytes() == old


def test_model_summary(tmp_path):
    folder = tmp_path / "BAAI--bge-small-en-v1.5"
    folder.mkdir()
    summary = manage_data.model_summary(str(folder))
    assert summary["name"] == "bge-small-en-v1.5" and summary["vendor"] == "BAAI" and summary["downloaded"] is True
    custom = manage_data.model_summary(str(tmp_path / "someone--custom-model"))
    assert custom["name"] == "someone--custom-model" and custom["known"] is False and custom["downloaded"] is False
    assert manage_data.model_summary("") is None


def test_unsafe_names_are_refused(root):
    outside = root / "Docs_for_DB"
    outside.mkdir()
    (outside / "keep.txt").write_text("x", encoding="utf-8")
    write_config(root, {"created_databases": {"../Docs_for_DB": entry()}})
    rows = manage_data.list_databases(manage_data.read_config())
    assert [(r["name"], r["folder"]) for r in rows] == [("../Docs_for_DB", False)]
    for bad in ("../Docs_for_DB", "..", ".", "", "a/b", "a\\b", "C:x"):
        for action in (manage_data.delete_database, manage_data.backup_database, manage_data.restore_database):
            with pytest.raises(ValueError):
                action(bad)
    assert (outside / "keep.txt").exists()
    assert "../Docs_for_DB" in read_config(root)["created_databases"]
