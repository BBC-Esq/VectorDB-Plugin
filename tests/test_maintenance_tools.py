import os
import subprocess
import sys
import zipfile
from pathlib import Path

PACKAGE_TABLE_SCRIPT = """
import importlib.metadata as metadata
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

app = QApplication([])
import tools.check_packages as checker


def rows(window):
    table = window.results_table
    return [tuple(table.item(row, column).text() if table.item(row, column) else None for column in range(3))
            for row in range(table.rowCount())]


installed = sorted(metadata.distributions(), key=lambda dist: dist.metadata["Name"].lower())
window = checker.PackageChecker()
window.show_message = lambda *args, **kwargs: None
first, last = installed[0], installed[-1]
window.on_outdated_packages_checked([(first.metadata["Name"], first.version, "999"), (last.metadata["Name"], last.version, "999")])
window.check_all_packages()
print("ALL", rows(window) == [(dist.metadata["Name"], dist.version, "N/A") for dist in installed])

window.on_outdated_packages_checked([("alpha", "1.0", "9.1"), ("bravo", "2.0", "9.2"), ("charlie", "3.0", "9.3")])
window.results_table.sortItems(1, Qt.AscendingOrder)
checker.importlib.metadata.version = lambda name: "2.5"
window.update_outdated_after_upgrade("alpha", "9.1")
print("UPGRADE", sorted(rows(window)) == [("alpha", "2.5", "9.1"), ("bravo", "2.0", "9.2"), ("charlie", "3.0", "9.3")])
sys.stdout.flush()
os._exit(0)
"""


def test_package_table_rows_keep_their_own_versions():
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", PACKAGE_TABLE_SCRIPT], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent.parent, env=environment, timeout=120)
    assert "ALL True" in result.stdout and "UPGRADE True" in result.stdout, result.stdout + result.stderr


def test_requirement_names_leave_out_versions_and_optional_extras():
    from tools.check_packages import required_names

    requirements = ["numpy>=1.22,<2.5", "Typing_Extensions (>=4.0)", "pytest; extra == 'dev'", "numpy<3", "not a requirement !"]
    assert required_names(requirements) == ["numpy", "typing-extensions"]
    assert required_names(None) == []


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
