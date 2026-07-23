import re

_MIN_INTERIOR_WORDS = 4
_MARGIN_FRAC = 0.125
_CORRUPT_THRESHOLD = 0.30
_IMAGE_COVER_FRAC = 0.5
_IMAGE_COVER_FRAC_NOTEXT = 0.05
_MIN_DRAWINGS = 20
_CID_RE = re.compile(r'\(cid:\d+\)')


def corrupt_fraction(text):
    s = text.strip()
    if not s:
        return 0.0
    bad = sum(1 for ch in s if ord(ch) == 0xFFFD or 0xE000 <= ord(ch) <= 0xF8FF)
    cid = sum(len(m) for m in _CID_RE.findall(s))
    return (bad + cid) / len(s)


def interior_word_count(page):
    r = page.rect
    mx, my = r.width * _MARGIN_FRAC, r.height * _MARGIN_FRAC
    x_lo, x_hi, y_lo, y_hi = r.x0 + mx, r.x1 - mx, r.y0 + my, r.y1 - my
    n = 0
    for w in page.get_text("words"):
        if not str(w[4]).strip():
            continue
        cx, cy = (w[0] + w[2]) / 2.0, (w[1] + w[3]) / 2.0
        if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
            n += 1
    return n


def has_page_image(page, frac=_IMAGE_COVER_FRAC):
    r = page.rect
    parea = r.width * r.height
    if parea <= 0:
        return False
    try:
        infos = page.get_image_info()
    except Exception:
        return False
    total = 0.0
    for im in infos:
        b = im.get("bbox")
        if b:
            total += abs((b[2] - b[0]) * (b[3] - b[1]))
    return total >= frac * parea


def has_visible_content(page, frac=_IMAGE_COVER_FRAC):
    if has_page_image(page, frac):
        return True
    try:
        return len(page.get_drawings()) >= _MIN_DRAWINGS
    except Exception:
        return False


def page_needs_ocr(page):
    text = page.get_text()
    if text.strip() and corrupt_fraction(text) > _CORRUPT_THRESHOLD:
        return True
    if interior_word_count(page) >= _MIN_INTERIOR_WORDS:
        return False
    frac = _IMAGE_COVER_FRAC if text.strip() else _IMAGE_COVER_FRAC_NOTEXT
    return has_visible_content(page, frac)


def document_needs_ocr(doc):
    if getattr(doc, 'needs_pass', False):
        return True
    for page in doc:
        if page_needs_ocr(page):
            return True
    return False
