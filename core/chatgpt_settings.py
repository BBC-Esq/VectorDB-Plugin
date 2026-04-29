AVAILABLE_OPENAI_MODELS = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
]

MODEL_DISPLAY_NAMES = {
    "gpt-5.5": "gpt-5.5 (Thinking)",
    "gpt-5.4": "gpt-5.4 (Thinking)",
    "gpt-5.4-mini": "gpt-5.4 mini",
}

MODEL_PRICING = {
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.25, 0.025, 2.00),
}

REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high", "xhigh"]
VERBOSITY_OPTIONS = ["low", "medium", "high"]

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
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
    return m.startswith("gpt-5.")


def supports_verbosity(model_name: str) -> bool:
    m = (model_name or "").strip().lower()
    return m.startswith("gpt-5.")


def migrate_legacy_model(model_name: str) -> str:
    if model_name in AVAILABLE_OPENAI_MODELS:
        return model_name
    return DEFAULT_OPENAI_MODEL
