import copy
import re

import requests
import yaml

from chat.kobold import KOBOLD_URL
from chat.minimax import MINIMAX_MODELS
from core.chatgpt_settings import (
    AVAILABLE_OPENAI_MODELS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_VERBOSITY,
    REASONING_EFFORT_OPTIONS,
    VERBOSITY_OPTIONS,
    get_display_name,
    get_model_pricing,
    supports_reasoning_effort,
    supports_verbosity,
)
from core.constants import PROJECT_ROOT
from core.utilities import save_config_atomically

BACKENDS = ["chatgpt", "lmstudio", "minimax", "kobold"]
TESTABLE = {"chatgpt", "lmstudio", "kobold"}
DEFAULT_CONNECTION_STR = "http://localhost:1234/v1"
PORT_RE = re.compile(r":(\d{1,5})(?=/)")
OPENAI_KEYS_URL = "https://platform.openai.com/api-keys"
MINIMAX_KEYS_URL = "https://platform.minimax.io/user-center/basic-information/interface-key"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
KEY_SECTIONS = {"chatgpt": "openai", "minimax": "minimax"}
SETTING_BACKEND = {
    "openai.api_key": "chatgpt",
    "openai.model": "chatgpt",
    "openai.verbosity": "chatgpt",
    "openai.reasoning_effort": "chatgpt",
    "server.port": "lmstudio",
    "server.show_thinking": "lmstudio",
    "minimax.api_key": "minimax",
}

HELP = {
    "openai.api_key": "Your OpenAI API key. ChatGPT answers are charged to the OpenAI account the key belongs to.",
    "openai.model": "The OpenAI model that writes the answers.",
    "openai.verbosity": "How long and detailed the answers are.",
    "openai.reasoning_effort": "How much the model reasons before it answers. More effort can help with hard questions "
                               "but is slower and costs more.",
    "openai.pricing": "What OpenAI charges per million tokens. Cached input is the discounted price for prompt text "
                      "OpenAI has seen recently.",
    "server.port": "Must match the port of LM Studio's local server, which is 1234 unless you changed it.",
    "server.show_thinking": "Shows a reasoning model's thinking in the answer. Models that don't reason are unaffected.",
    "minimax.api_key": "Your MiniMax API key. MiniMax answers are charged to the MiniMax account the key belongs to.",
    "minimax.models": "The MiniMax models this program can use.",
    "kobold.address": "The address of KoboldCpp's local server.",
}


class ConfigError(Exception):
    pass


class InvalidValue(Exception):
    pass


def config_file():
    return PROJECT_ROOT / "config.yaml"


def load_config():
    try:
        with open(config_file(), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"config.yaml could not be read: {e}")
    return data if isinstance(data, dict) else {}


def save_config(cfg):
    save_config_atomically(cfg, config_file(), allow_unicode=True)


def _section(cfg, name):
    if not isinstance(cfg.get(name), dict):
        cfg[name] = {}
    return cfg[name]


def _existing(cfg, name):
    value = cfg.get(name)
    return value if isinstance(value, dict) else {}


def _text(value):
    return "" if value is None else str(value).strip()


def key_hint(key):
    key = _text(key)
    return key[-4:] if len(key) >= 12 else ""


def port_of(connection):
    match = PORT_RE.search(connection or "")
    return match.group(1) if match else ""


def with_port(connection, port):
    match = PORT_RE.search(connection)
    return connection[:match.start(1)] + str(port) + connection[match.end(1):]


def openai_values(cfg):
    section = _existing(cfg, "openai")
    model = section.get("model") if section.get("model") in AVAILABLE_OPENAI_MODELS else DEFAULT_OPENAI_MODEL
    verbosity = section.get("verbosity") if section.get("verbosity") in VERBOSITY_OPTIONS else DEFAULT_VERBOSITY
    reasoning = section.get("reasoning_effort") if section.get("reasoning_effort") in REASONING_EFFORT_OPTIONS else DEFAULT_REASONING_EFFORT
    return section, model, verbosity, reasoning


def normalize_config():
    if not config_file().exists():
        return False
    try:
        cfg = load_config()
    except ConfigError:
        return False
    section = cfg.get("openai")
    if not isinstance(section, dict):
        return False
    original = copy.deepcopy(section)
    _, model, verbosity, reasoning = openai_values(cfg)
    for name, value in (("model", model), ("verbosity", verbosity), ("reasoning_effort", reasoning)):
        if section.get(name) and section[name] != value:
            section[name] = value
    if section == original:
        return False
    save_config(cfg)
    return True


def price_note(model):
    input_cost, _, output_cost = get_model_pricing(model)
    return f"${input_cost:.2f} in · ${output_cost:.2f} out"


def build_state(selected="chatgpt", tests=None):
    tests = tests or {}
    error = None
    try:
        cfg = load_config()
    except ConfigError as e:
        cfg, error = {}, str(e)
    openai, model, verbosity, reasoning = openai_values(cfg)
    server = _existing(cfg, "server")
    connection = _text(server.get("connection_str"))
    minimax = _existing(cfg, "minimax")
    input_cost, cached_cost, output_cost = get_model_pricing(model)
    return {
        "selected": selected if selected in BACKENDS else BACKENDS[0],
        "error": error,
        "help": HELP,
        "chatgpt": {
            "key_set": bool(_text(openai.get("api_key"))),
            "key_hint": key_hint(openai.get("api_key")),
            "keys_url": OPENAI_KEYS_URL,
            "model": model,
            "models": [{"value": m, "label": get_display_name(m), "note": price_note(m)} for m in AVAILABLE_OPENAI_MODELS],
            "pricing": {"input": input_cost, "cached": cached_cost, "output": output_cost},
            "verbosity": verbosity,
            "verbosities": VERBOSITY_OPTIONS,
            "show_verbosity": supports_verbosity(model),
            "reasoning": reasoning,
            "reasonings": REASONING_EFFORT_OPTIONS,
            "show_reasoning": supports_reasoning_effort(model),
            "test": tests.get("chatgpt"),
        },
        "lmstudio": {
            "connection": connection,
            "port": port_of(connection),
            "missing": not connection,
            "malformed": bool(connection) and not port_of(connection),
            "show_thinking": bool(server.get("show_thinking", False)),
            "default_connection": DEFAULT_CONNECTION_STR,
            "default_port": port_of(DEFAULT_CONNECTION_STR),
            "test": tests.get("lmstudio"),
        },
        "minimax": {
            "key_set": bool(_text(minimax.get("api_key"))),
            "key_hint": key_hint(minimax.get("api_key")),
            "keys_url": MINIMAX_KEYS_URL,
            "models": MINIMAX_MODELS,
        },
        "kobold": {
            "address": KOBOLD_URL,
            "test": tests.get("kobold"),
        },
    }


def _key_setter(section_name):
    def setter(cfg, value):
        key = _text(value)
        if any(c.isspace() for c in key):
            raise InvalidValue("An API key can't contain spaces.")
        section = _section(cfg, section_name)
        if _text(section.get("api_key")) == key:
            return False
        section["api_key"] = key or None
        return True
    return setter


def _choice_setter(name, options, label):
    def setter(cfg, value):
        if value not in options:
            raise InvalidValue(f"{label} must be one of: {', '.join(options)}.")
        section = _section(cfg, "openai")
        if section.get(name) == value:
            return False
        section[name] = value
        return True
    return setter


def _set_port(cfg, value):
    text = _text(value)
    if not re.fullmatch(r"\d{1,5}", text) or not 1 <= int(text) <= 65535:
        raise InvalidValue("Port must be a number between 1 and 65535.")
    server = _section(cfg, "server")
    connection = _text(server.get("connection_str"))
    if not connection:
        new = with_port(DEFAULT_CONNECTION_STR, int(text))
    elif not PORT_RE.search(connection):
        raise InvalidValue(f"The saved LM Studio address ({connection}) has no port to change. Fix it in config.yaml.")
    else:
        new = with_port(connection, int(text))
    if new == server.get("connection_str"):
        return False
    server["connection_str"] = new
    return True


def _set_show_thinking(cfg, value):
    server = _section(cfg, "server")
    value = bool(value)
    if bool(server.get("show_thinking", False)) == value:
        return False
    server["show_thinking"] = value
    return True


HANDLERS = {
    "openai.api_key": _key_setter("openai"),
    "openai.model": _choice_setter("model", AVAILABLE_OPENAI_MODELS, "Model"),
    "openai.verbosity": _choice_setter("verbosity", VERBOSITY_OPTIONS, "Verbosity"),
    "openai.reasoning_effort": _choice_setter("reasoning_effort", REASONING_EFFORT_OPTIONS, "Reasoning effort"),
    "server.port": _set_port,
    "server.show_thinking": _set_show_thinking,
    "minimax.api_key": _key_setter("minimax"),
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
    except (InvalidValue, ConfigError) as e:
        return {"ok": False, "error": str(e)}
    except OSError as e:
        return {"ok": False, "error": f"The setting could not be saved: {e}"}
    return {"ok": True, "changed": bool(changed)}


def saved_key(backend):
    section = KEY_SECTIONS.get(backend)
    if section is None:
        return ""
    return _text(_existing(load_config(), section).get("api_key"))


def _names(items, limit=3):
    shown = ", ".join(items[:limit])
    return f"{shown} and {len(items) - limit} more" if len(items) > limit else shown


def check_chatgpt(cfg):
    openai, model, _, _ = openai_values(cfg)
    key = _text(openai.get("api_key"))
    if not key:
        return {"status": "error", "message": "Enter an API key first."}
    try:
        response = requests.get(OPENAI_MODELS_URL, headers={"Authorization": f"Bearer {key}"}, timeout=15)
    except requests.RequestException:
        return {"status": "error", "message": "Could not reach OpenAI. Check your internet connection and try again."}
    if response.status_code == 401:
        return {"status": "error", "message": "OpenAI did not accept this API key. Check that it was copied completely."}
    if response.status_code == 429:
        return {"status": "warn", "message": "OpenAI accepted the key but is limiting its requests right now. Try again in a minute."}
    if response.status_code != 200:
        return {"status": "error", "message": f"OpenAI answered with an error (HTTP {response.status_code}). Try again later."}
    try:
        ids = {item.get("id") for item in response.json().get("data", [])}
    except ValueError:
        ids = set()
    name = get_display_name(model)
    if ids and model not in ids:
        return {"status": "warn", "message": f"The key works, but OpenAI doesn't offer {name} to this account. Choose another model."}
    return {"status": "ok", "message": f"The key works and can use {name}."}


def check_lmstudio(cfg):
    connection = _text(_existing(cfg, "server").get("connection_str"))
    if not connection:
        return {"status": "error", "message": "Enter LM Studio's port first."}
    try:
        response = requests.get(connection.rstrip("/") + "/models", timeout=5)
    except requests.RequestException:
        return {"status": "error", "message": f"LM Studio isn't answering at {connection}. Start its local server, load a model and try again."}
    try:
        models = [item.get("id") for item in response.json().get("data", []) if item.get("id")] if response.status_code == 200 else None
    except (ValueError, AttributeError):
        models = None
    if models is None:
        return {"status": "error", "message": f"Something answered at {connection}, but it doesn't look like LM Studio (HTTP {response.status_code})."}
    if not models:
        return {"status": "warn", "message": "LM Studio's server is running, but no model is loaded. Load a model in LM Studio."}
    return {"status": "ok", "message": f"LM Studio is running with {_names(models)}."}


def check_kobold(cfg):
    try:
        response = requests.get(f"{KOBOLD_URL}/api/v1/model", timeout=5)
    except requests.RequestException:
        return {"status": "error", "message": f"KoboldCpp isn't answering at {KOBOLD_URL}. Start KoboldCpp with a model and try again."}
    try:
        name = _text(response.json().get("result")) if response.status_code == 200 else None
    except (ValueError, AttributeError):
        name = None
    if name is None:
        return {"status": "error", "message": f"Something answered at {KOBOLD_URL}, but it doesn't look like KoboldCpp (HTTP {response.status_code})."}
    name = name.split("/", 1)[1] if name.startswith("koboldcpp/") else name
    return {"status": "ok", "message": f"KoboldCpp is running {name}." if name else "KoboldCpp is running."}


CHECKS = {"chatgpt": check_chatgpt, "lmstudio": check_lmstudio, "kobold": check_kobold}


def check_connection(backend, cfg):
    return CHECKS[backend](cfg)
