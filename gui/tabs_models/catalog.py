import re
from functools import lru_cache
from pathlib import Path

import torch
import yaml

from core.constants import PROJECT_ROOT, VECTOR_MODELS
from core.utilities import (
    cuda_usable,
    get_appropriate_dtype,
    get_embedding_dtype_and_batch,
    runs_on_this_hardware,
)
from gui.download_model import is_complete_download

BENCHMARKS = [
    {
        "key": "english_retrieval",
        "label": "English",
        "title": "English retrieval",
        "about": "Retrieval benchmark score on English text.",
    },
    {
        "key": "multilingual_retrieval",
        "label": "Multilingual",
        "title": "Multilingual retrieval",
        "about": "Retrieval benchmark score across many languages.",
    },
    {
        "key": "rteb_legal",
        "label": "Legal",
        "title": "Legal retrieval (RTEB)",
        "about": "Score on the legal portion of the RTEB retrieval benchmark.",
    },
    {
        "key": "mteb_code",
        "label": "Code",
        "title": "Code (MTEB)",
        "about": "Score on the MTEB code benchmark.",
    },
    {
        "key": "code_information_retrieval",
        "label": "Code IR",
        "title": "Code information retrieval",
        "about": "Score on code information retrieval tasks.",
    },
]

LICENSES = {
    "mit": ("MIT", "MIT License"),
    "apache-2.0": ("Apache-2.0", "Apache License 2.0"),
    "cc0-1.0": ("CC0-1.0", "Creative Commons Zero (public domain)"),
    "gemma - commercial ok": ("Gemma", "Gemma Terms of Use; commercial use is allowed"),
}

CPU_NOTE = (
    "Running on the CPU (no supported NVIDIA GPU). Smaller models create databases much faster. "
    "Approximate time per 10,000 chunks on a 24-core CPU: small models ~5 min, base and 300M models "
    "~7-19 min, large models ~20-25 min, 0.6B models ~28-40 min, the 1.7B model ~95 min. "
    "Slower CPUs take longer."
)


def model_directory_name(model):
    return model.get("cache_dir") or model["repo_id"].replace("/", "--")


def is_downloaded(model):
    return is_complete_download(Path("Models") / "vector" / model_directory_name(model))


def find_model(repo_id):
    for vendor_models in VECTOR_MODELS.values():
        for model in vendor_models:
            if model["repo_id"] == repo_id:
                return model
    return None


def read_precision_settings():
    try:
        with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        config = {}
    compute = config.get("Compute_Device") or {}
    database = config.get("database") or {}
    return str(compute.get("database_creation") or "cpu").lower(), bool(database.get("half", False))


def parse_parameters(text):
    match = re.fullmatch(r"\s*([\d.]+)\s*([kmb]?)\s*", str(text).lower())
    if not match:
        return 0.0
    scale = {"k": 0.001, "m": 1.0, "b": 1000.0, "": 1.0}[match.group(2)]
    return float(match.group(1)) * scale


@lru_cache(maxsize=1)
def detect_hardware():
    if not cuda_usable():
        return {"cuda": False, "gpu": None, "gpu_short": None, "cc": None}
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    short = re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", name).strip() or name
    return {"cuda": True, "gpu": name, "gpu_short": short, "cc": f"{major}.{minor}"}


def _dtype_name(dtype):
    return str(dtype).split(".")[-1]


def runtime_precision(model, device):
    native = model.get("precision", "float32")
    model_path = model_directory_name(model)
    half_off, _ = get_embedding_dtype_and_batch(device, False, native, model_path, True)
    half_on, _ = get_embedding_dtype_and_batch(device, True, native, model_path, True)
    unguarded = get_appropriate_dtype(device, True, native)
    return {
        "native": native,
        "half_off": _dtype_name(half_off),
        "half_on": _dtype_name(half_on),
        "fp16_guarded": bool(unguarded == torch.float16 and half_on == torch.float32),
    }


def build_payload(downloading=None):
    hardware = dict(detect_hardware())
    creation_device, half = read_precision_settings()
    cuda = hardware["cuda"]
    device = "cuda" if cuda and creation_device == "cuda" else "cpu"
    use_half = half and cuda
    hardware.update({"device": device, "half": use_half, "cpu_only": not cuda})

    models = []
    for vendor, vendor_models in VECTOR_MODELS.items():
        for model in vendor_models:
            if not runs_on_this_hardware(model):
                continue
            precision = runtime_precision(model, device)
            precision["current"] = precision["half_on"] if use_half else precision["half_off"]
            license_key = str(model.get("license", "")).strip()
            license_label, license_title = LICENSES.get(
                license_key.lower(), (license_key.upper() or "Unknown", license_key)
            )
            scores = {}
            for bench in BENCHMARKS:
                value = model.get(bench["key"])
                scores[bench["key"]] = float(value) if isinstance(value, (int, float)) else None
            models.append({
                "id": model["repo_id"],
                "order": len(models),
                "vendor": vendor,
                "name": model["name"],
                "repo_id": model["repo_id"],
                "url": f"https://huggingface.co/{model['repo_id']}",
                "dimensions": int(model["dimensions"]),
                "max_sequence": int(model["max_sequence"]),
                "size_mb": int(model["size_mb"]),
                "parameters_m": parse_parameters(model.get("parameters", "")),
                "precision": precision,
                "license": {"key": license_key, "label": license_label, "title": license_title},
                "requires_cuda": bool(model.get("requires_cuda", False)),
                "custom_code": bool(model.get("custom_code", False)),
                "downloaded": is_downloaded(model),
                "scores": scores,
            })

    benchmarks = []
    for bench in BENCHMARKS:
        scored = sorted(
            ((m["scores"][bench["key"]], m["id"]) for m in models if m["scores"][bench["key"]] is not None),
            key=lambda pair: -pair[0],
        )
        benchmarks.append({
            **bench,
            "count": len(scored),
            "min": scored[-1][0] if scored else None,
            "max": scored[0][0] if scored else None,
            "ranking": [model_id for _, model_id in scored],
        })

    return {
        "hardware": hardware,
        "cpu_note": None if cuda else CPU_NOTE,
        "benchmarks": benchmarks,
        "models": models,
        "downloading": downloading,
    }
