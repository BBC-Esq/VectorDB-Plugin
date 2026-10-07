import copy
import re
from functools import lru_cache
from pathlib import Path

import torch
import yaml

from core.constants import KOKORO_SPEEDS, KOKORO_VOICES, TTS_BACKENDS, VECTOR_MODELS, VISION_MODELS, WHISPER_SPEECH_MODELS
from core.utilities import cuda_usable, fallback_if_unavailable, runs_on_this_hardware, save_config_atomically

CONFIG_FILE = Path("config.yaml")

PIPELINE_PRESETS = ["minimal", "low", "normal", "high", "maximum"]
DOCUMENT_TYPES = [("", "All files"), ("document", "Documents"), ("image", "Images"), ("audio", "Audio")]
DEVICE_LABELS = {"cpu": "CPU", "cuda": "CUDA", "mps": "MPS"}
DEFAULT_VISION_MODEL = "Liquid-VL - 480M"

WHISPER_SPEECH_SPEAKERS = ["default", "classic", "voice_b"]
WHISPER_SPEECH_VOICE_CLONING_LABEL = "Voice Cloning (Coming Soon)"

KYUTAI_POCKET_VOICES = [
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette",
    "eponine", "eve", "fantine", "george", "jane", "javert", "jean", "marius",
    "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera",
]

KYUTAI_MODELS = {
    "1.6B (EN+FR, ~4.2GB VRAM)": ("kyutai/tts-1.6b-en_fr", 32),
    "0.75B (EN, ~2GB VRAM)": ("kyutai/tts-0.75b-en-public", 16),
}

KYUTAI_VOICES = {
    "Default Male": "expresso/ex04-ex03_default_002_channel2_239s.wav",
    "Fast Male 1": "expresso/ex01-ex02_fast_001_channel1_104s.wav",
    "Fast Female": "expresso/ex01-ex02_fast_001_channel2_73s.wav",
    "Fast Male 2": "expresso/ex04-ex03_fast_001_channel2_25s.wav",
    "Happy Male": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "Happy Female 1": "expresso/ex04-ex02_happy_001_channel1_118s.wav",
    "Happy Female 2": "expresso/ex04-ex02_happy_001_channel2_140s.wav",
    "Enunciated Female": "expresso/ex04-ex03_enunciated_001_channel2_342s.wav",
}

TTS_SPECS = {
    "bark": {
        "label": "Bark",
        "extras": {
            "size": {"label": "Model", "options": ["normal", "small"], "default": "small"},
            "speaker": {
                "label": "Speaker",
                "options": [f"v2/en_speaker_{i}" for i in range(10)],
                "default": "v2/en_speaker_6",
            },
        },
    },
    "whisperspeech": {
        "label": "WhisperSpeech",
        "extras": {
            "s2a": {
                "label": "S2A model",
                "options": list(WHISPER_SPEECH_MODELS["s2a"]),
                "default": list(WHISPER_SPEECH_MODELS["s2a"])[0],
            },
            "t2s": {
                "label": "T2S model",
                "options": list(WHISPER_SPEECH_MODELS["t2s"]),
                "default": list(WHISPER_SPEECH_MODELS["t2s"])[0],
            },
            "speaker": {
                "label": "Speaker",
                "options": WHISPER_SPEECH_SPEAKERS + [WHISPER_SPEECH_VOICE_CLONING_LABEL],
                "disabled": [WHISPER_SPEECH_VOICE_CLONING_LABEL],
                "default": WHISPER_SPEECH_SPEAKERS[0],
            },
        },
    },
    "chattts": {"label": "ChatTTS", "extras": {}},
    "chatterbox": {"label": "Chatterbox", "extras": {}},
    "googletts": {"label": "Google TTS", "extras": {}},
    "kokoro": {
        "label": "Kokoro",
        "extras": {
            "voice": {"label": "Voice", "options": list(KOKORO_VOICES), "default": "bm_george"},
            "speed": {"label": "Speed", "options": list(KOKORO_SPEEDS), "default": "Medium"},
        },
    },
    "kyutaipocket": {
        "label": "Kyutai Pocket",
        "extras": {
            "voice": {"label": "Voice", "options": KYUTAI_POCKET_VOICES, "default": "alba"},
        },
        "toggles": {
            "quantize": {
                "label": "Quantize (int8)",
                "default": True,
                "help": (
                    "Apply int8 quantization. The developers claim no loss of quality (WER unchanged) "
                    "with ~48% less RAM and ~27% faster inference. Feel free to test and decide for yourself."
                ),
            },
        },
    },
    "kyutai": {
        "label": "Kyutai",
        "extras": {
            "model": {"label": "Model", "options": list(KYUTAI_MODELS), "default": "1.6B (EN+FR, ~4.2GB VRAM)"},
            "voice": {"label": "Voice", "options": list(KYUTAI_VOICES), "default": "Happy Male"},
        },
    },
}

TTS_NOTES = {
    "bark": "GPU",
    "whisperspeech": "GPU",
    "chattts": "CPU/GPU",
    "chatterbox": "CPU/GPU",
    "googletts": "CPU, online",
    "kokoro": "CPU",
    "kyutaipocket": "CPU",
    "kyutai": "GPU",
}

HELP = {
    "query.device": "Device used when searching a database. CPU is recommended so the GPU's memory stays free for the chat model.",
    "query.similarity": (
        "Minimum relevance (0-1) a chunk needs to be returned. Higher returns fewer, more relevant chunks; "
        "lower returns more. Don't use 1."
    ),
    "query.contexts": "Maximum number of chunks (aka contexts) to return.",
    "query.search_term": (
        "Removes chunks that do not contain this term as a case-insensitive substring. "
        "Leave it empty to turn the filter off."
    ),
    "query.document_types": "Only allows chunks that originate from certain file types.",
    "create.device": "Device used to create databases. Use CUDA if you have a supported NVIDIA GPU.",
    "create.pipeline_preset": (
        "Controls CPU parallelism during database creation. Minimal: sequential processing (1 thread/process). "
        "Low: light parallelism (2-4 workers). Normal: moderate parallelism (default). High: aggressive parallelism. "
        "Maximum: all available CPU cores."
    ),
    "create.half": "Uses bfloat16/float16 for about a 2x speedup. Requires a supported NVIDIA GPU.",
    "create.half.unavailable": (
        "Half-precision requires a supported NVIDIA GPU. Databases are created in full precision on this computer."
    ),
    "create.chunk_size": (
        "Upper limit (in characters, not tokens) that a chunk can be after being split. Make sure it fits within "
        "the Max Sequence of the embedding model, which is measured in tokens; roughly 3-4 characters = 1 token."
    ),
    "create.chunk_overlap": "Characters shared between neighboring chunks. Set to 25-50% of chunk size.",
    "tts.backend": "The voice used when a response is read aloud. Backends that need a GPU are hidden on CPU-only computers.",
    "vision.model": "The model that describes images when they are added to a database.",
}


MESSAGES = {
    "query.similarity": "Similarity must be a number between 0.0 and 1.0.",
    "query.contexts": "Contexts must be a whole number from 1 to 1,000.",
    "create.chunk_size": "Chunk size must be a whole number from 1 to 100,000.",
    "create.chunk_overlap": "Chunk overlap must be a whole number from 0 to 99,999.",
}


class SettingsError(Exception):
    pass


def load_config():
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return {}
    except (OSError, yaml.YAMLError) as e:
        raise SettingsError(f"config.yaml could not be read: {e}")
    return data if isinstance(data, dict) else {}


def save_config(cfg):
    save_config_atomically(cfg, CONFIG_FILE, allow_unicode=True)


def _section(cfg, name):
    if not isinstance(cfg.get(name), dict):
        cfg[name] = {}
    return cfg[name]


def _pick(value, options, default):
    return value if value in options else default


def _devices(cfg):
    available = (cfg.get("Compute_Device") or {}).get("available") or ["cpu"]
    return [str(device) for device in available]


def available_backends():
    return [key for key in TTS_SPECS if runs_on_this_hardware(TTS_BACKENDS.get(key))]


def current_backend(cfg):
    backend = fallback_if_unavailable((cfg.get("tts") or {}).get("model", "googletts"), TTS_BACKENDS, "googletts")
    options = available_backends()
    return backend if backend in options else options[0]


def available_vision_models():
    return [name for name, info in VISION_MODELS.items() if runs_on_this_hardware(info)]


def current_vision_model(cfg):
    return fallback_if_unavailable((cfg.get("vision") or {}).get("chosen_model"), VISION_MODELS, DEFAULT_VISION_MODEL)


def _model_key_for_file(models, filename):
    for key, (file, _size) in models.items():
        if file == filename:
            return key
    return next(iter(models))


def tts_values(cfg):
    tts_cfg = cfg.get("tts") or {}
    bark_cfg = cfg.get("bark") or {}
    pocket_cfg = cfg.get("kyutaipocket") or {}
    kokoro_cfg = cfg.get("kokoro") or {}
    kyutai_cfg = cfg.get("kyutai") or {}
    spec = {key: value["extras"] for key, value in TTS_SPECS.items()}

    try:
        speed = float(kokoro_cfg.get("speed", KOKORO_SPEEDS["Medium"]))
    except (TypeError, ValueError):
        speed = KOKORO_SPEEDS["Medium"]
    speed_label = next((label for label, value in KOKORO_SPEEDS.items() if abs(value - speed) < 1e-6), "Medium")

    return {
        "bark": {
            "size": _pick(bark_cfg.get("size", "small"), spec["bark"]["size"]["options"], "small"),
            "speaker": _pick(bark_cfg.get("speaker", "v2/en_speaker_6"), spec["bark"]["speaker"]["options"], "v2/en_speaker_6"),
        },
        "whisperspeech": {
            "s2a": _model_key_for_file(WHISPER_SPEECH_MODELS["s2a"], tts_cfg.get("s2a")),
            "t2s": _model_key_for_file(WHISPER_SPEECH_MODELS["t2s"], tts_cfg.get("t2s")),
            "speaker": _pick(tts_cfg.get("speaker"), WHISPER_SPEECH_SPEAKERS, WHISPER_SPEECH_SPEAKERS[0]),
        },
        "chattts": {},
        "chatterbox": {},
        "googletts": {},
        "kokoro": {
            "voice": _pick(kokoro_cfg.get("voice", "bm_george"), KOKORO_VOICES, "bm_george"),
            "speed": speed_label,
        },
        "kyutaipocket": {
            "voice": _pick(pocket_cfg.get("voice", "alba"), KYUTAI_POCKET_VOICES, "alba"),
            "quantize": bool(pocket_cfg.get("quantize", True)),
        },
        "kyutai": {
            "model": _pick(kyutai_cfg.get("model_display_name"), KYUTAI_MODELS, "1.6B (EN+FR, ~4.2GB VRAM)"),
            "voice": _pick(kyutai_cfg.get("voice_display_name"), KYUTAI_VOICES, "Happy Male"),
        },
    }


def apply_tts(cfg, backend, values):
    tts_cfg = _section(cfg, "tts")
    tts_cfg["model"] = backend
    if backend == "bark":
        bark = _section(cfg, "bark")
        bark["size"] = values["size"]
        bark["speaker"] = values["speaker"]
    elif backend == "whisperspeech":
        tts_cfg["s2a"] = WHISPER_SPEECH_MODELS["s2a"][values["s2a"]][0]
        tts_cfg["t2s"] = WHISPER_SPEECH_MODELS["t2s"][values["t2s"]][0]
        if values["speaker"] in WHISPER_SPEECH_SPEAKERS:
            tts_cfg["speaker"] = values["speaker"]
    elif backend == "kyutaipocket":
        pocket = _section(cfg, "kyutaipocket")
        pocket["language"] = "english"
        pocket["voice"] = values["voice"]
        pocket["quantize"] = bool(values["quantize"])
        pocket["temp"] = 0.7
    elif backend == "kokoro":
        kokoro = _section(cfg, "kokoro")
        kokoro["voice"] = values["voice"]
        kokoro["speed"] = KOKORO_SPEEDS.get(values["speed"], KOKORO_SPEEDS["Medium"])
    elif backend == "kyutai":
        kyutai = _section(cfg, "kyutai")
        hf_repo, n_q = KYUTAI_MODELS[values["model"]]
        kyutai["model_display_name"] = values["model"]
        kyutai["hf_repo"] = hf_repo
        kyutai["n_q"] = n_q
        kyutai["voice"] = KYUTAI_VOICES[values["voice"]]
        kyutai["voice_display_name"] = values["voice"]
        kyutai["temp"] = 0.6
        kyutai["cfg_coef"] = 2.0


def normalize_config():
    if not CONFIG_FILE.exists():
        return False
    try:
        cfg = load_config()
    except SettingsError:
        return False
    original = copy.deepcopy(cfg)
    backend = current_backend(cfg)
    apply_tts(cfg, backend, tts_values(cfg)[backend])
    _section(cfg, "vision")["chosen_model"] = current_vision_model(cfg)
    if cfg == original:
        return False
    save_config(cfg)
    return True


def _embedding_model(cfg):
    path = cfg.get("EMBEDDING_MODEL_NAME")
    if not path:
        return None
    folder = Path(str(path)).name
    for vendor_models in VECTOR_MODELS.values():
        for model in vendor_models:
            if model.get("cache_dir") == folder:
                return {"name": model["name"], "max_sequence": int(model["max_sequence"])}
    return None


@lru_cache(maxsize=1)
def gpu_name():
    if not cuda_usable():
        return None
    name = torch.cuda.get_device_name(0)
    return re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", name).strip() or name


def _device_options(cfg):
    return [{"value": device, "label": DEVICE_LABELS.get(device, device.upper())} for device in _devices(cfg)]


def _sections(cfg, cuda):
    devices = _device_options(cfg)
    return [
        {
            "id": "query",
            "title": "Database Query",
            "icon": "search",
            "fields": [
                {"key": "query.device", "label": "Device", "type": "segmented", "options": devices},
                {"key": "query.similarity", "label": "Similarity", "type": "number", "decimals": 2,
                 "min": 0, "max": 1, "step": 0.05, "placeholder": "0.0 - 1.0", "error": MESSAGES["query.similarity"]},
                {"key": "query.contexts", "label": "Contexts", "type": "number", "decimals": 0,
                 "min": 1, "max": 1000, "step": 1, "unit": "chunks", "error": MESSAGES["query.contexts"]},
                {"key": "query.search_term", "label": "Search term filter", "type": "text",
                 "placeholder": "No filter", "span": 2},
                {"key": "query.document_types", "label": "File type", "type": "segmented", "span": 2,
                 "options": [{"value": value, "label": label} for value, label in DOCUMENT_TYPES]},
            ],
        },
        {
            "id": "create",
            "title": "Database Creation",
            "icon": "build",
            "fields": [
                {"key": "create.device", "label": "Device", "type": "segmented", "options": devices},
                {"key": "create.chunk_size", "label": "Chunk size", "type": "number", "decimals": 0,
                 "min": 1, "max": 100000, "step": 50, "unit": "characters", "error": MESSAGES["create.chunk_size"]},
                {"key": "create.chunk_overlap", "label": "Chunk overlap", "type": "number", "decimals": 0,
                 "min": 0, "max": 99999, "step": 25, "unit": "characters", "error": MESSAGES["create.chunk_overlap"]},
                {"key": "create.half", "label": "Half precision", "type": "toggle", "text": "About 2x faster",
                 "disabled": not cuda},
                {"key": "create.pipeline_preset", "label": "Pipeline performance", "type": "segmented", "span": 3,
                 "options": [{"value": preset, "label": preset.capitalize()} for preset in PIPELINE_PRESETS]},
            ],
        },
        {"id": "tts", "title": "Text to Speech", "icon": "speaker", "custom": "tts"},
        {"id": "vision", "title": "Vision Model", "icon": "eye", "custom": "vision"},
    ]


def _tts_state(cfg):
    backend = current_backend(cfg)
    values = tts_values(cfg)
    backends = []
    for key in available_backends():
        spec = TTS_SPECS[key]
        extras = []
        for name, meta in spec["extras"].items():
            visible = not (key == "kyutai" and name == "voice" and not values["kyutai"]["model"].startswith("1.6B"))
            extras.append({
                "name": name,
                "label": meta["label"],
                "value": values[key][name],
                "visible": visible,
                "options": [
                    {"value": option, "label": option, "disabled": option in meta.get("disabled", [])}
                    for option in meta["options"]
                ],
            })
        toggles = [
            {"name": name, "label": meta["label"], "value": bool(values[key].get(name, meta["default"])), "help": meta["help"]}
            for name, meta in spec.get("toggles", {}).items()
        ]
        backends.append({"key": key, "label": spec["label"], "note": TTS_NOTES.get(key, ""), "extras": extras, "toggles": toggles})
    return {"current": backend, "backends": backends}


def _vision_state(cfg):
    models = []
    for name in available_vision_models():
        info = VISION_MODELS[name]
        models.append({
            "name": name,
            "size": str(info.get("size", "")),
            "vram": str(info.get("vram", "")),
            "speed": info.get("characters_per_second"),
            "avg_length": info.get("avg_length"),
            "vision_component": str(info.get("vision_component", "")),
            "chat_component": str(info.get("chat_component", "")),
            "precision": str(info.get("precision", "")),
            "quant": str(info.get("quant", "")),
            "license": str(info.get("license", "")),
        })
    return {"current": current_vision_model(cfg), "models": models}


def build_state():
    error = None
    try:
        cfg = load_config()
    except SettingsError as e:
        cfg, error = {}, str(e)
    database = cfg.get("database") or {}
    compute = cfg.get("Compute_Device") or {}
    devices = _devices(cfg)
    cuda = cuda_usable()
    values = {
        "query.device": compute.get("database_query") if compute.get("database_query") in devices else devices[0],
        "query.similarity": database.get("similarity"),
        "query.contexts": database.get("contexts"),
        "query.search_term": str(database.get("search_term") or ""),
        "query.document_types": str(database.get("document_types") or ""),
        "create.device": compute.get("database_creation") if compute.get("database_creation") in devices else devices[0],
        "create.chunk_size": database.get("chunk_size"),
        "create.chunk_overlap": database.get("chunk_overlap"),
        "create.half": bool(database.get("half", False)) and cuda,
        "create.pipeline_preset": _pick(database.get("pipeline_preset", "normal"), PIPELINE_PRESETS, "normal"),
    }
    return {
        "error": error,
        "cuda": cuda,
        "gpu": gpu_name() if cuda else None,
        "values": values,
        "sections": _sections(cfg, cuda),
        "help": HELP,
        "tts": _tts_state(cfg),
        "vision": _vision_state(cfg),
        "embedding": _embedding_model(cfg),
    }


def _whole_number(value, low, high, message):
    text = str(value).strip().replace(",", "").replace(" ", "").replace("_", "")
    try:
        number = int(text)
    except ValueError:
        raise SettingsError(message)
    if not low <= number <= high:
        raise SettingsError(message)
    return number


def _set(cfg, section, key, value):
    target = _section(cfg, section)
    if target.get(key) == value and key in target:
        return False
    target[key] = value
    return True


def _set_device(cfg, key, value):
    if value not in _devices(cfg):
        raise SettingsError(f"Choose one of: {', '.join(_devices(cfg))}.")
    return _set(cfg, "Compute_Device", key, value)


def _set_similarity(cfg, value):
    try:
        number = float(str(value).strip())
    except ValueError:
        raise SettingsError(MESSAGES["query.similarity"])
    if not 0.0 <= number <= 1.0:
        raise SettingsError(MESSAGES["query.similarity"])
    return _set(cfg, "database", "similarity", number)


def _set_contexts(cfg, value):
    number = _whole_number(value, 1, 1000, MESSAGES["query.contexts"])
    return _set(cfg, "database", "contexts", number)


def _set_search_term(cfg, value):
    return _set(cfg, "database", "search_term", str(value or "").strip())


def _set_document_types(cfg, value):
    if value not in [key for key, _label in DOCUMENT_TYPES]:
        raise SettingsError("Choose a file type from the list.")
    return _set(cfg, "database", "document_types", value)


def _set_pipeline_preset(cfg, value):
    if value not in PIPELINE_PRESETS:
        raise SettingsError("Choose a pipeline performance level from the list.")
    return _set(cfg, "database", "pipeline_preset", value)


def _set_half(cfg, value):
    if not cuda_usable():
        raise SettingsError(HELP["create.half.unavailable"])
    return _set(cfg, "database", "half", bool(value))


def _set_chunk_size(cfg, value):
    size = _whole_number(value, 1, 100000, MESSAGES["create.chunk_size"])
    overlap = (cfg.get("database") or {}).get("chunk_overlap", 0)
    if isinstance(overlap, int) and overlap >= size:
        raise SettingsError(f"Chunk overlap must be less than chunk size. Lower the overlap ({overlap:,}) first.")
    return _set(cfg, "database", "chunk_size", size)


def _set_chunk_overlap(cfg, value):
    overlap = _whole_number(value, 0, 99999, MESSAGES["create.chunk_overlap"])
    size = (cfg.get("database") or {}).get("chunk_size", 0)
    if isinstance(size, int) and size and overlap >= size:
        raise SettingsError(f"Chunk overlap must be less than chunk size ({size:,}).")
    return _set(cfg, "database", "chunk_overlap", overlap)


def _set_tts_backend(cfg, value):
    if value not in available_backends():
        raise SettingsError("That text to speech backend is not available on this computer.")
    before = copy.deepcopy(cfg)
    apply_tts(cfg, value, tts_values(cfg)[value])
    return cfg != before


def _set_tts_option(cfg, value):
    backend = value.get("backend")
    name = value.get("name")
    choice = value.get("value")
    if backend not in available_backends() or backend != current_backend(cfg):
        raise SettingsError("Select that text to speech backend first.")
    spec = TTS_SPECS[backend]
    values = tts_values(cfg)[backend]
    if name in spec["extras"]:
        meta = spec["extras"][name]
        if choice not in meta["options"] or choice in meta.get("disabled", []):
            raise SettingsError(f"Choose a {meta['label'].lower()} from the list.")
        values[name] = choice
    elif name in spec.get("toggles", {}):
        values[name] = bool(choice)
    else:
        raise SettingsError("Unknown text to speech option.")
    before = copy.deepcopy(cfg)
    apply_tts(cfg, backend, values)
    return cfg != before


def _set_vision_model(cfg, value):
    if value not in available_vision_models():
        raise SettingsError("That vision model is not available on this computer.")
    return _set(cfg, "vision", "chosen_model", value)


HANDLERS = {
    "query.device": lambda cfg, value: _set_device(cfg, "database_query", value),
    "query.similarity": _set_similarity,
    "query.contexts": _set_contexts,
    "query.search_term": _set_search_term,
    "query.document_types": _set_document_types,
    "create.device": lambda cfg, value: _set_device(cfg, "database_creation", value),
    "create.chunk_size": _set_chunk_size,
    "create.chunk_overlap": _set_chunk_overlap,
    "create.half": _set_half,
    "create.pipeline_preset": _set_pipeline_preset,
    "tts.backend": _set_tts_backend,
    "tts.option": _set_tts_option,
    "vision.model": _set_vision_model,
}


def apply_change(key, value):
    handler = HANDLERS.get(key)
    if handler is None:
        return {"ok": False, "error": f"Unknown setting: {key}"}
    try:
        cfg = load_config()
        changed = handler(cfg, value)
        if changed:
            save_config(cfg)
    except SettingsError as e:
        return {"ok": False, "error": str(e), "state": build_state()}
    except OSError as e:
        return {"ok": False, "error": f"The settings could not be saved: {e}", "state": build_state()}
    return {"ok": True, "changed": bool(changed), "state": build_state()}
