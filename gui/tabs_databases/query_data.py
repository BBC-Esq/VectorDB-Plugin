import re
from pathlib import Path

import yaml

from core.constants import CHAT_MODELS, PROJECT_ROOT
from core.utilities import cuda_usable, runs_on_this_hardware, save_config_atomically
from gui.tabs_databases import manage_data
from gui.tabs_settings.settings_state import DOCUMENT_TYPES, TTS_SPECS

BACKENDS = ["Local Model", "Kobold", "LM Studio", "ChatGPT", "MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"]
MINIMAX_BACKENDS = {"MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed"}
SETTINGS_DIALOG_TAB = {"ChatGPT": 0, "LM Studio": 1, "Kobold": 3}
SETTINGS_LABEL = "Open Chat Backend Settings"
DOCUMENT_TYPE_LABELS = dict(DOCUMENT_TYPES)

ITEM = re.compile(r"<li>(.*?)</li>", re.S)
LINK = re.compile(r'<a href="file:(?P<path>[^"]*)"[^>]*>(?P<name>.*?)</a>', re.S)
SCORE = re.compile(r"\[<span[^>]*>(?P<low>-?[0-9.]+)(?:-(?P<high>-?[0-9.]+))?</span>\]")
PAGES = re.compile(r"p\.(?P<pages>[0-9 ,\-]+)</span>")
TAGS = re.compile(r"<[^>]+>")
TOKENS = re.compile(
    r"available tokens \((?P<available>\d+)\).*?rag instruction \((?P<instruction>\d+)\).*?query \((?P<question>\d+)\)"
    r".*?contexts \((?P<contexts>\d+)\).*?response \((?P<response>\d+)\).*?=\s*(?P<remaining>-?\d+) remaining", re.S)


def config_path():
    return PROJECT_ROOT / "config.yaml"


def load_config():
    try:
        with open(config_path(), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def queryable_databases(cfg):
    rows = []
    for entry in manage_data.list_databases(cfg):
        if not entry["registered"] or not entry["folder"]:
            continue
        model = manage_data.model_summary(entry["model_path"])
        rows.append({
            "name": entry["name"],
            "model": model["name"] if model else None,
            "model_missing": bool(model) and not model["downloaded"],
        })
    return rows


def remembered_database(cfg, names):
    wanted = str((cfg.get("database") or {}).get("database_to_search") or "")
    if wanted in names:
        return wanted
    return names[0] if names else None


def remember_database(name):
    try:
        with open(config_path(), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(cfg, dict):
        return False
    database = cfg.setdefault("database", {})
    if not isinstance(database, dict) or database.get("database_to_search") == name:
        return False
    database["database_to_search"] = name
    save_config_atomically(cfg, config_path(), allow_unicode=True)
    return True


def query_settings(cfg):
    database = cfg.get("database") or {}
    compute = cfg.get("Compute_Device") or {}
    document_types = str(database.get("document_types") or "")
    return {
        "contexts": database.get("contexts"),
        "similarity": database.get("similarity"),
        "search_term": str(database.get("search_term") or ""),
        "document_types": DOCUMENT_TYPE_LABELS.get(document_types, document_types or "All files"),
        "device": compute.get("database_query") or "cpu",
    }


def tts_label(cfg):
    key = str((cfg.get("tts") or {}).get("model") or "").lower()
    return TTS_SPECS.get(key, {}).get("label") or key


def chat_model_downloaded(info):
    folder = PROJECT_ROOT / "Models" / "chat" / info["cache_dir"]
    try:
        return folder.is_dir() and any(folder.iterdir())
    except OSError:
        return False


def local_models(cfg):
    has_token = bool(str(cfg.get("hf_access_token") or "").strip())
    gpu = cuda_usable()
    rows = []
    for name, info in CHAT_MODELS.items():
        if not runs_on_this_hardware(info):
            continue
        rows.append({
            "name": name,
            "memory": round(info["vram"] / 1024, 1) if gpu and info.get("vram") else None,
            "downloaded": chat_model_downloaded(info),
            "needs_token": bool(info.get("gated")) and not has_token,
        })
    return rows


def readiness(cfg, backend, local_model, models):
    if backend == "ChatGPT" and not str((cfg.get("openai") or {}).get("api_key") or "").strip():
        return {"message": "ChatGPT needs an OpenAI API key.", "action": "settings", "label": SETTINGS_LABEL}
    if backend in MINIMAX_BACKENDS and not str((cfg.get("minimax") or {}).get("api_key") or "").strip():
        return {"message": "MiniMax needs an API key.", "action": "minimax_key", "label": "Enter the MiniMax API key"}
    if backend == "LM Studio" and not str((cfg.get("server") or {}).get("connection_str") or "").strip():
        return {"message": "LM Studio needs its server address.", "action": "settings", "label": SETTINGS_LABEL}
    if backend == "Local Model":
        model = next((m for m in models if m["name"] == local_model), None)
        if model is None:
            return {"message": "Choose a local model." if models else "No local chat model runs on this computer.", "action": None, "label": ""}
        if model["needs_token"]:
            return {"message": f"{local_model} requires a Hugging Face access token.", "action": "hf_token", "label": "Enter a Hugging Face access token"}
    return None


def parse_citations(html):
    citations = []
    for item in ITEM.findall(html or ""):
        link = LINK.search(item)
        if not link:
            continue
        score = SCORE.search(item)
        pages = PAGES.search(item)
        low = float(score.group("low")) if score else None
        high = float(score.group("high")) if score and score.group("high") else low
        path = link.group("path")
        citations.append({
            "name": TAGS.sub("", link.group("name")) or Path(path).name,
            "path": path,
            "low": low,
            "high": high,
            "pages": pages.group("pages").strip() if pages else "",
        })
    citations.sort(key=lambda c: (-(c["high"] if c["high"] is not None else -1), c["name"].lower()))
    return citations


def parse_token_counts(html):
    match = TOKENS.search(TAGS.sub("", html or ""))
    if not match:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def citations_text(citations):
    lines = []
    for c in citations:
        score = "" if c["low"] is None else (f"{c['low']:.4f}" if c["low"] == c["high"] else f"{c['low']:.4f}-{c['high']:.4f}")
        pages = f" p.{c['pages']}" if c["pages"] else ""
        lines.append(f"{c['name']} [{score}]{pages}".replace(" []", ""))
    return lines


def turn_text(turn):
    if turn["kind"] == "chunks":
        parts = []
        for i, chunk in enumerate(turn.get("chunks") or [], start=1):
            page = f" (page {chunk['page']})" if chunk.get("page") else ""
            parts.append(f"CONTEXT {i} | {chunk['name']}{page}\n{chunk['text']}")
        return "\n\n".join(parts)
    text = (turn.get("answer") or "").strip()
    lines = citations_text(turn.get("citations") or [])
    if lines:
        text += "\n\nCitations:\n" + "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))
    return text
