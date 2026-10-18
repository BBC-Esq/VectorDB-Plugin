AVAILABLE_OPENAI_MODELS = [
    "gpt-6-astra",
    "gpt-6.1-sol",
    "gpt-6-luna",
]

MODEL_DISPLAY_NAMES = {
    "gpt-6-astra": "gpt-6 Astra (flagship)",
    "gpt-6.1-sol": "gpt-6.1 Sol (balanced)",
    "gpt-6-luna": "gpt-6 Luna (economy)",
}

MODEL_PRICING = {
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}

REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high", "xhigh", "max"]
VERBOSITY_OPTIONS = ["low", "medium", "high"]

MAX_EFFORT_MODEL_PREFIXES = ("gpt-5.6", "gpt-6")
NO_NONE_EFFORT_MODELS = {"gpt-6-astra", "gpt-6.1-sol"}
REASONING_MODEL_PREFIXES = ("gpt-5.", "gpt-6")

DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_VERBOSITY = "low"
DEFAULT_REASONING_EFFORT = "medium"


def get_display_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def get_model_from_display_name(display_name: str) -> str:
    for model, name in MODEL_DISPLAY_NAMES.items():
        if name == display_name:
            return model
    return display_name


def get_model_pricing(model_name: str) -> tuple[float, float, float]:
    return MODEL_PRICING.get(model_name, (0.00, 0.00, 0.00))


def supports_reasoning_effort(model_name: str) -> bool:
    m = (model_name or "").strip().lower()
    if m.endswith("-chat-latest"):
        return False
    return m.startswith(REASONING_MODEL_PREFIXES)


def supports_verbosity(model_name: str) -> bool:
    m = (model_name or "").strip().lower()
    return m.startswith(REASONING_MODEL_PREFIXES)


def supports_max_effort(model_name: str) -> bool:
    m = (model_name or "").strip().lower()
    return m.startswith(MAX_EFFORT_MODEL_PREFIXES)


def supports_none_effort(model_name: str) -> bool:
    m = (model_name or "").strip().lower()
    return m not in NO_NONE_EFFORT_MODELS


def reasoning_effort_options(model_name: str) -> list:
    if not supports_reasoning_effort(model_name):
        return []
    options = list(REASONING_EFFORT_OPTIONS)
    if not supports_max_effort(model_name):
        options.remove("max")
    if not supports_none_effort(model_name):
        options.remove("none")
    return options


def clamp_reasoning_effort(model_name: str, effort: str) -> str:
    allowed = reasoning_effort_options(model_name)
    if not allowed or effort in allowed:
        return effort
    if effort == "max":
        return "xhigh"
    if effort == "none":
        return "low"
    return DEFAULT_REASONING_EFFORT


def migrate_legacy_model(model_name: str) -> str:
    if model_name in AVAILABLE_OPENAI_MODELS:
        return model_name
    return DEFAULT_OPENAI_MODEL
