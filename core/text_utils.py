"""Qt-free, torch-free text helpers, split out of core.utilities so headless
split/extract workers can import normalize_text without loading PySide6 or torch
(which pull conflicting OpenMP runtimes). Do not add GUI/torch imports here."""

import re


def normalize_text(text, preserve_whitespace=False):
    import unicodedata

    if text is None:
        return None

    if isinstance(text, (list, tuple)):
        text = " ".join(str(item) for item in text if item is not None)

    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)

    INVISIBLE_CHARS = {
        chr(0x00ad), chr(0x200b), chr(0x200c), chr(0x200d), chr(0x200e), chr(0x200f),
        chr(0x2060), chr(0x2061), chr(0x2062), chr(0x2063), chr(0x2064), chr(0xfeff),
    }

    cleaned = []
    for char in text:
        code = ord(char)
        if char == '\n' or char == '\t':
            if preserve_whitespace:
                cleaned.append(char)
            else:
                cleaned.append(' ')
        elif char == '\r':
            cleaned.append(' ')
        elif code < 32:
            continue
        elif code == 127:
            continue
        elif code > 65535:
            continue
        elif char in INVISIBLE_CHARS:
            continue
        elif 128 <= code <= 159:
            continue
        elif code == 65533:
            continue
        elif 57344 <= code <= 63743:
            continue
        else:
            cleaned.append(char)

    result = "".join(cleaned)

    if preserve_whitespace:
        result = re.sub(r'[^\S\n\t]+', ' ', result)
        result = re.sub(r' *\n *', '\n', result)
        result = re.sub(r'\n{3,}', '\n\n', result)
    else:
        result = " ".join(result.split())

    result = result.strip()
    return result if result else None
