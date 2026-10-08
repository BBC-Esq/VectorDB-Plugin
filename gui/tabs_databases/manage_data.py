import json
import os
import shutil
import sqlite3
import stat
import time
from contextlib import closing
from pathlib import Path

import yaml

from core.constants import PROJECT_ROOT
from core.utilities import save_config_atomically
from gui.download_model import is_complete_download
from gui.tabs_databases.create_config import model_info

JEEVES_DATABASE = "user_manual"

KINDS = [
    ("pdf", "PDF", {".pdf"}),
    ("word", "Word", {".docx", ".doc"}),
    ("text", "Text", {".txt", ".md", ".rtf"}),
    ("web", "Web pages", {".html", ".htm"}),
    ("email", "Email", {".eml", ".msg"}),
    ("sheet", "Spreadsheets", {".csv", ".xls", ".xlsx", ".xlsm"}),
    ("image", "Images", {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}),
    ("audio", "Audio", {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".aiff", ".aif",
                        ".webm", ".mp4", ".mkv", ".mov", ".avi"}),
]
KIND_BY_SUFFIX = {suffix: kind for kind, _, suffixes in KINDS for suffix in suffixes}


class ConfigError(Exception):
    pass


def config_path():
    return PROJECT_ROOT / "config.yaml"


def vector_root():
    return PROJECT_ROOT / "Vector_DB"


def backup_root():
    return PROJECT_ROOT / "Vector_DB_Backup"


def kind_of(name):
    return KIND_BY_SUFFIX.get(Path(name).suffix.lower(), "other")


def read_config():
    try:
        with open(config_path(), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError("config.yaml is missing.")
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"config.yaml could not be read: {e}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("config.yaml does not contain settings.")
    return data


def registered_databases(cfg):
    entries = cfg.get("created_databases")
    if not isinstance(entries, dict):
        return {}
    return {str(key): (value if isinstance(value, dict) else {}) for key, value in entries.items()
            if str(key) != JEEVES_DATABASE}


def folder_names(root):
    try:
        with os.scandir(root) as it:
            return {entry.name for entry in it
                    if entry.is_dir() and entry.name != JEEVES_DATABASE and not entry.name.startswith(".")}
    except OSError:
        return set()


def list_databases(cfg):
    registered = registered_databases(cfg)
    folders = folder_names(vector_root())
    backups = folder_names(backup_root())
    rows = []
    for name in sorted(set(registered) | folders, key=lambda n: (n.lower(), n)):
        entry = registered.get(name)
        rows.append({
            "name": name,
            "registered": entry is not None,
            "folder": name in folders,
            "backup": name in backups,
            "model_path": str(entry.get("model") or "") if entry else "",
            "chunk_size": entry.get("chunk_size") if entry else None,
            "chunk_overlap": entry.get("chunk_overlap") if entry else None,
        })
    return rows


def model_summary(model_path):
    if not model_path:
        return None
    path = Path(model_path)
    vendor, info = model_info(path.name)
    return {
        "folder": path.name,
        "path": str(path),
        "name": info["name"] if info else path.name,
        "vendor": vendor or "",
        "known": info is not None,
        "downloaded": is_complete_download(path),
    }


def signature(name):
    folder = vector_root() / name
    parts = []
    for path in (folder, folder / "metadata.db", folder / "index_metadata.json"):
        try:
            st = path.stat()
            parts.append((st.st_mtime_ns, st.st_size))
        except OSError:
            parts.append(None)
    return tuple(parts)


def folder_size(path):
    total = 0
    stack = [str(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        else:
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def connect_readonly(db_path, should_stop=None):
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    if should_stop is not None:
        conn.set_progress_handler(lambda: 1 if should_stop() else 0, 20000)
    return conn


def read_index_metadata(folder):
    try:
        with open(Path(folder) / "index_metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def database_info(name, should_stop=None):
    folder = vector_root() / name
    meta = read_index_metadata(folder)
    info = {
        "files": None,
        "chunks": meta.get("num_vectors"),
        "dimensions": meta.get("dimensions"),
        "size": folder_size(folder),
        "created": None,
        "complete": False,
        "problem": None,
    }
    db_path = folder / "metadata.db"
    try:
        info["created"] = db_path.stat().st_mtime
    except OSError:
        try:
            info["created"] = folder.stat().st_ctime
        except OSError:
            pass
        info["problem"] = "The database's file list (metadata.db) is missing, so the database is incomplete."
        return info
    try:
        with closing(connect_readonly(db_path, should_stop)) as conn:
            info["files"] = conn.execute("SELECT COUNT(*) FROM document_metadata").fetchone()[0]
            if info["chunks"] is None:
                info["chunks"] = conn.execute("SELECT COUNT(*) FROM hash_chunk_ids").fetchone()[0]
    except sqlite3.Error as e:
        info["problem"] = f"The database's file list could not be read: {e}"
        return info
    if not meta:
        info["problem"] = "The database's index information (index_metadata.json) is missing, so the database is incomplete."
        return info
    info["complete"] = True
    return info


def file_rows(name, should_stop=None):
    db_path = vector_root() / name / "metadata.db"
    with closing(connect_readonly(db_path, should_stop)) as conn:
        docs = conn.execute("SELECT file_name, file_path, hash FROM document_metadata").fetchall()
        counts = dict(conn.execute("SELECT hash, COUNT(*) FROM hash_chunk_ids GROUP BY hash").fetchall())
    sharing = {}
    for _, _, digest in docs:
        sharing[digest] = sharing.get(digest, 0) + 1
    rows = []
    for file_name, file_path, digest in docs:
        path = str(file_path or "")
        label = str(file_name or "") or (Path(path).name if path else "Unnamed file")
        rows.append((label, path, round(counts.get(digest, 0) / max(1, sharing.get(digest, 1)))))
    rows.sort(key=lambda r: (r[0].lower(), r[1].lower()))
    return rows


def compact_rows(rows):
    dirs = {}
    out = []
    for name, path, chunks in rows:
        parent, base = os.path.split(path)
        row = [name, kind_of(path or name), chunks]
        if path and base == name:
            row.append(dirs.setdefault(parent, len(dirs)))
        else:
            row.append(-1)
            row.append(path)
        out.append(row)
    return {"dirs": list(dirs), "rows": out}


def missing_rows(rows, should_stop):
    missing = []
    for index, (_, path, _) in enumerate(rows):
        if should_stop():
            return None
        if not path or not os.path.exists(path):
            missing.append(index)
    return missing


def is_link(path):
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def remove_tree(path):
    path = Path(path)
    if not os.path.lexists(path):
        return None
    if is_link(path):
        return f"{path} is a link to another location, so it was not deleted."
    for attempt in range(4):
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.lexists(path):
            return None
        time.sleep(0.3 * (attempt + 1))
    return f"{path} could not be fully deleted. A program may be using its files."


def check_name(name):
    if not name or name in (".", "..") or Path(name).name != name or any(c in name for c in ("/", "\\", ":")):
        raise ValueError(f"'{name}' is not a valid database name, so nothing was changed.")


def delete_database(name):
    check_name(name)
    cfg = read_config()
    entries = cfg.get("created_databases")
    changed = False
    if isinstance(entries, dict):
        for key in [k for k in entries if str(k) == name]:
            del entries[key]
            changed = True
    database = cfg.get("database")
    if isinstance(database, dict) and database.get("database_to_search") == name:
        database["database_to_search"] = ""
        changed = True
    if changed:
        save_config_atomically(cfg, config_path(), allow_unicode=True)
    failed = [problem for problem in (remove_tree(vector_root() / name), remove_tree(backup_root() / name)) if problem]
    return {"failed": failed}


def copy_into_place(source, target, temp):
    problem = remove_tree(temp)
    if problem:
        raise OSError(problem)
    try:
        shutil.copytree(source, temp)
    except Exception:
        remove_tree(temp)
        raise
    if os.path.lexists(target):
        problem = remove_tree(target)
        if problem:
            remove_tree(temp)
            raise OSError(problem)
    os.replace(temp, target)


def backup_database(name):
    check_name(name)
    source = vector_root() / name
    if not (source / "metadata.db").exists():
        raise FileNotFoundError(f"{source} is not a complete database, so it was not backed up.")
    backup_root().mkdir(parents=True, exist_ok=True)
    copy_into_place(source, backup_root() / name, PROJECT_ROOT / f".backup_{name}")


def restore_database(name):
    check_name(name)
    source = backup_root() / name
    target = vector_root() / name
    if os.path.lexists(target):
        raise FileExistsError(f"{target} already exists.")
    if not (source / "metadata.db").exists():
        raise FileNotFoundError(f"The backup copy in {source} is not a complete database.")
    vector_root().mkdir(parents=True, exist_ok=True)
    copy_into_place(source, target, PROJECT_ROOT / f".restore_{name}")
