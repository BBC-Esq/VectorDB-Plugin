import json
from pathlib import Path

import yaml

from core.constants import PIPELINE_PRESETS, PROJECT_ROOT, VECTOR_MODELS, VISION_MODELS
from core.utilities import cuda_usable, max_database_name_length, runs_on_this_hardware, save_config_atomically
from gui.download_model import is_complete_download
from gui.tabs_models.catalog import runtime_precision

RESERVED_NAMES = {"null", "none"}


def config_path():
    return PROJECT_ROOT / "config.yaml"


def models_dir():
    return PROJECT_ROOT / "Models" / "vector"


def load_config():
    try:
        with open(config_path(), "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def model_info(folder_name):
    wanted = folder_name.lower()
    for vendor, models in VECTOR_MODELS.items():
        for model in models:
            if (model.get("cache_dir") or "").lower() == wanted:
                return vendor, model
    return None, None


def available_models():
    folder = models_dir()
    if not folder.exists():
        return []
    gpu_only = {(m.get("cache_dir") or "").lower() for models in VECTOR_MODELS.values() for m in models
                if not runs_on_this_hardware(m)}
    rows = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not is_complete_download(path) or path.name.lower() in gpu_only:
            continue
        vendor, info = model_info(path.name)
        rows.append({
            "path": str(path),
            "folder": path.name,
            "name": info["name"] if info else path.name,
            "vendor": vendor or "",
            "dimensions": info.get("dimensions") if info else None,
            "max_sequence": info.get("max_sequence") if info else None,
            "known": info is not None,
        })
    return rows


def embedding_dimensions(selected_path):
    if "stella" in selected_path.lower() or "static-retrieval" in selected_path.lower():
        return 1024
    config_json_path = Path(selected_path) / "config.json"
    if config_json_path.exists():
        with open(config_json_path, 'r', encoding='utf-8') as json_file:
            model_config = json.load(json_file)
        embedding_dimensions = model_config.get("hidden_size") or model_config.get("d_model")
        if embedding_dimensions and isinstance(embedding_dimensions, int):
            return embedding_dimensions
    return None


def select_model(selected_path):
    config_data = {}
    if config_path().exists():
        with open(config_path(), 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file) or {}
    if selected_path:
        config_data["EMBEDDING_MODEL_NAME"] = selected_path
        dimensions = embedding_dimensions(selected_path)
        if dimensions:
            config_data["EMBEDDING_MODEL_DIMENSIONS"] = dimensions
    else:
        config_data.pop("EMBEDDING_MODEL_NAME", None)
        config_data.pop("EMBEDDING_MODEL_DIMENSIONS", None)
    save_config_atomically(config_data, config_path(), allow_unicode=True)


def creation_device(cfg):
    device = (cfg.get("Compute_Device") or {}).get("database_creation") or "cpu"
    return device if device != "cuda" or cuda_usable() else "cpu"


def precision_for(cfg):
    model_path = cfg.get("EMBEDDING_MODEL_NAME")
    if not model_path:
        return None
    _, info = model_info(Path(model_path).name)
    if not info:
        return "unknown"
    try:
        precision = runtime_precision(info, creation_device(cfg))
    except Exception:
        return info.get("precision", "float32")
    half = bool((cfg.get("database") or {}).get("half", False))
    return precision["half_on"] if half else precision["half_off"]


def gpu_available_but_cpu_chosen(cfg):
    compute = cfg.get("Compute_Device") or {}
    available = compute.get("available") or []
    return compute.get("database_creation") == "cpu" and any(d in available for d in ("cuda", "mps"))


def existing_databases():
    try:
        return sorted(p.name for p in (PROJECT_ROOT / "Vector_DB").iterdir() if p.is_dir())
    except OSError:
        return []


def name_limit():
    return max(3, max_database_name_length(PROJECT_ROOT))


def settings_summary(cfg):
    db = cfg.get("database") or {}
    preset = db.get("pipeline_preset") or "normal"
    vision = (cfg.get("vision") or {}).get("chosen_model")
    vision_info = VISION_MODELS.get(vision)
    return {
        "chunk_size": db.get("chunk_size"),
        "chunk_overlap": db.get("chunk_overlap"),
        "half": bool(db.get("half", False)),
        "device": creation_device(cfg),
        "preset": preset if preset in PIPELINE_PRESETS else "normal",
        "precision": precision_for(cfg),
        "vision": vision,
        "vision_needs_gpu": bool(vision_info) and not runs_on_this_hardware(vision_info),
        "cpu_warning": gpu_available_but_cpu_chosen(cfg),
    }
