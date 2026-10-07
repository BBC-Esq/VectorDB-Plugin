from PySide6.QtGui import QColor

from core.constants import THEMES


def _luminance(color):
    def channel(value):
        value /= 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(color.red()) + 0.7152 * channel(color.green()) + 0.0722 * channel(color.blue())


def _contrast(a, b):
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _readable_on(background, preferred, fallbacks):
    if _contrast(preferred, background) >= 2.0:
        return preferred
    return max([preferred, *fallbacks], key=lambda c: _contrast(c, background))


def theme_payload(theme_name):
    colors = THEMES.get(theme_name) or THEMES["default"]
    q = {key: QColor(value) for key, value in colors.items()}
    light = _luminance(q["bg_window"]) > 0.4
    fallbacks = [q["selection_fg"], QColor("black"), QColor("white")]
    return {
        "scheme": "light" if light else "dark",
        "colors": {key: color.name() for key, color in q.items()},
        "accent": "#00806b" if light else "#00bf9e",
        "gold": "#9a6700" if light else "#e3b341",
        "danger": "#c0392b" if light else "#ff6b6b",
        "on_control": _readable_on(q["bg_control"], q["text_primary"], fallbacks).name(),
        "on_control_hover": _readable_on(q["bg_control_hover"], q["text_primary"], fallbacks).name(),
    }
