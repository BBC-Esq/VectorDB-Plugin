import re

_MIN_INTERIOR_WORDS = 4
_MARGIN_FRAC = 0.125
_CORRUPT_THRESHOLD = 0.30
_IMAGE_COVER_FRAC = 0.5
_SPARSE_INTERIOR_WORDS = 100
_SPARSE_TEXT_LINES = 3.5
_IMAGE_COVER_FRAC_NOTEXT = 0.05
_MIN_DRAWINGS = 20
_CID_RE = re.compile(r'\(cid:\d+\)')
_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
_WORD_SPLIT_RE = re.compile(r'[ \t\n\r\xa0]+')
OCR_PRODUCER_PREFIX = "VectorDB-Plugin"


def corrupt_fraction(text):
    s = text.strip()
    if not s:
        return 0.0
    bad = sum(1 for ch in s if ord(ch) == 0xFFFD or 0xE000 <= ord(ch) <= 0xF8FF)
    cid = sum(len(m) for m in _CID_RE.findall(s))
    return (bad + cid) / len(s)


def control_fraction(text):
    words = [w for w in _WORD_SPLIT_RE.split(text) if w]
    total = sum(len(w) for w in words)
    if not total:
        return 0.0
    return sum(len(w) for w in words if _CONTROL_RE.search(w)) / total


def text_is_garbled(text):
    return bool(text.strip()) and (corrupt_fraction(text) > _CORRUPT_THRESHOLD
                                   or control_fraction(text) > _CORRUPT_THRESHOLD)


def has_ocr_layer(page):
    try:
        return any(span.get("type") == 3 for span in page.get_texttrace())
    except Exception:
        return False


def produced_by_ocr(doc):
    return str((doc.metadata or {}).get('producer') or '').startswith(OCR_PRODUCER_PREFIX)


def word_rects(page):
    rect_cls = type(page.rect)
    try:
        matrix = page.rotation_matrix
    except Exception:
        matrix = None
    out = []
    for w in page.get_text("words"):
        if not str(w[4]).strip():
            continue
        r = rect_cls(w[0], w[1], w[2], w[3])
        if matrix is not None:
            r = r * matrix
        r.normalize()
        out.append(r)
    return out


def interior_words(page):
    r = page.rect
    mx, my = r.width * _MARGIN_FRAC, r.height * _MARGIN_FRAC
    x_lo, x_hi, y_lo, y_hi = r.x0 + mx, r.x1 - mx, r.y0 + my, r.y1 - my
    out = []
    for wr in word_rects(page):
        cx, cy = (wr.x0 + wr.x1) / 2.0, (wr.y0 + wr.y1) / 2.0
        if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
            out.append(wr)
    return out


def interior_word_count(page):
    return len(interior_words(page))


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


def text_line_equivalents(page, rects=None):
    ws = interior_words(page) if rects is None else rects
    if not ws:
        return 0.0
    merged = []
    for y0, y1 in sorted((r.y0, r.y1) for r in ws):
        if merged and y0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], y1)
        else:
            merged.append([y0, y1])
    covered = sum(b - a for a, b in merged)
    heights = sorted(r.y1 - r.y0 for r in ws)
    line = heights[len(heights) // 2]
    if line <= 0:
        return 0.0
    return covered / line


def scanned_text_is_sparse(page, rects=None):
    if not has_page_image(page, _IMAGE_COVER_FRAC):
        return False
    ws = interior_words(page) if rects is None else rects
    if len(ws) > _SPARSE_INTERIOR_WORDS:
        return False
    return text_line_equivalents(page, ws) <= _SPARSE_TEXT_LINES


def page_needs_ocr(page):
    text = page.get_text()
    if text_is_garbled(text):
        return True
    if has_ocr_layer(page):
        return False
    rects = interior_words(page)
    if scanned_text_is_sparse(page, rects):
        return True
    if len(rects) >= _MIN_INTERIOR_WORDS:
        return False
    frac = _IMAGE_COVER_FRAC if text.strip() else _IMAGE_COVER_FRAC_NOTEXT
    return has_visible_content(page, frac)


def document_needs_ocr(doc):
    if getattr(doc, 'needs_pass', False):
        return True
    our_own = produced_by_ocr(doc)
    for page in doc:
        if not page_needs_ocr(page):
            continue
        if our_own and not text_is_garbled(page.get_text()):
            continue
        return True
    return False
