import hashlib
import logging
from pathlib import Path

import fitz
import numpy as np

from core.constants import PROJECT_ROOT
from core.pdf_ocr_gate import page_needs_ocr, produced_by_ocr, text_is_garbled
from core.text_order import column_reading_order

logger = logging.getLogger(__name__)

CHECKS = {"detector": "detector-1", "quick": "quick-2"}
READERS = {"rapidocr": "rapidocr-1", "tesseract": "tesseract-1"}
READER_LABELS = {"rapidocr": "RapidOCR", "tesseract": "Tesseract"}

GATE_DPI = 200
RAPIDOCR_DPI = 300
TESSERACT_DPI = 400
TEXT_SCORE = 0.3
TESSERACT_ORIENT_CONF = 3.0
MAX_PIXELS = 30_000_000
ORIENT_ACCEPT_CONF = 0.90
ORIENT_MARGIN = 1.20
DETECT_ONLY = dict(use_det=True, use_cls=False, use_rec=False)
READ = dict(use_det=True, use_cls=False, use_rec=True)

MODEL_SOURCES = {
    "PP-OCRv6_det_small.onnx": ("PaddlePaddle/PP-OCRv6_small_det_onnx", "inference.onnx"),
    "PP-OCRv6_det_medium.onnx": ("PaddlePaddle/PP-OCRv6_medium_det_onnx", "inference.onnx"),
    "PP-OCRv6_rec_small.onnx": ("PaddlePaddle/PP-OCRv6_small_rec_onnx", "inference.onnx"),
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": ("SWHL/RapidOCR", "PP-OCRv1/ch_ppocr_mobile_v2.0_cls_infer.onnx"),
}
MODEL_SHA256 = {
    "PP-OCRv6_det_small.onnx": "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e",
    "PP-OCRv6_det_medium.onnx": "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1",
    "PP-OCRv6_rec_small.onnx": "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634",
    "PP-OCRv6_rec_keys.txt": "f7aa897ca828a4c7c9e2739c30f9161a33306d532f020bcdb91dcfb664a5507e",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
}


def models_dir():
    return Path(PROJECT_ROOT) / "Models" / "ocr" / "rapidocr"


def _model_is_intact(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == MODEL_SHA256[path.name]


def _fetch_model(name, path):
    from huggingface_hub import hf_hub_download
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "PP-OCRv6_rec_keys.txt":
        import yaml
        settings = hf_hub_download("PaddlePaddle/PP-OCRv6_small_rec_onnx", "inference.yml")
        characters = yaml.safe_load(Path(settings).read_text(encoding="utf-8"))["PostProcess"]["character_dict"]
        path.write_bytes("\n".join(characters).encode("utf-8"))
        return
    repo, filename = MODEL_SOURCES[name]
    print(f"Downloading the OCR model {name}", flush=True)
    path.write_bytes(Path(hf_hub_download(repo, filename)).read_bytes())


def ensure_model(name):
    path = models_dir() / name
    if path.exists():
        if _model_is_intact(path):
            return path
        path.unlink()
    _fetch_model(name, path)
    if not _model_is_intact(path):
        path.unlink(missing_ok=True)
        raise RuntimeError(f"The OCR model {name} did not download correctly.")
    return path


def ensure_models(check, reader):
    names = ["PP-OCRv6_rec_small.onnx", "PP-OCRv6_rec_keys.txt", "ch_ppocr_mobile_v2.0_cls_mobile.onnx"]
    if check == "detector":
        names.append("PP-OCRv6_det_small.onnx")
    if reader == "rapidocr":
        names.append("PP-OCRv6_det_medium.onnx")
    if check == "detector" or reader == "rapidocr":
        for name in names:
            ensure_model(name)


def _only_warnings(record):
    return record.levelno >= logging.WARNING


def rapidocr_engine(detector, threads):
    from rapidocr import ModelType, OCRVersion, RapidOCR
    params = {
        "Det.model_path": str(ensure_model(f"PP-OCRv6_det_{detector}.onnx")),
        "Det.ocr_version": OCRVersion.PPOCRV6,
        "Rec.model_path": str(ensure_model("PP-OCRv6_rec_small.onnx")),
        "Rec.rec_keys_path": str(ensure_model("PP-OCRv6_rec_keys.txt")),
        "Cls.model_path": str(ensure_model("ch_ppocr_mobile_v2.0_cls_mobile.onnx")),
        "EngineConfig.onnxruntime.intra_op_num_threads": threads,
    }
    if detector == "medium":
        params["Det.model_type"] = ModelType.MEDIUM
    if _only_warnings not in logging.getLogger("RapidOCR").filters:
        logging.getLogger("RapidOCR").addFilter(_only_warnings)
    return RapidOCR(params=params)


def tesseract_engine():
    import tesserocr
    data = str(Path(PROJECT_ROOT) / "share" / "tessdata")
    reader = tesserocr.PyTessBaseAPI(lang="eng", path=data, psm=tesserocr.PSM.AUTO, oem=tesserocr.OEM.DEFAULT)
    try:
        orientation = tesserocr.PyTessBaseAPI(lang="osd", path=data, psm=tesserocr.PSM.OSD_ONLY)
    except Exception:
        orientation = None
    return reader, orientation


def _dpi_for(page, dpi):
    inches = max(page.rect.width, 1.0) * max(page.rect.height, 1.0) / 5184.0
    return max(36, int(min(dpi, (MAX_PIXELS / inches) ** 0.5)))


def render(page, dpi):
    pixmap = page.get_pixmap(dpi=_dpi_for(page, dpi))
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n)[:, :, :3][:, :, ::-1].copy()


def visible_character_centres(page):
    points = []
    for span in page.get_texttrace():
        if span.get("type") == 3 or not span.get("opacity", 1):
            continue
        for character in span.get("chars") or ():
            if chr(character[0]).isspace():
                continue
            box = character[3]
            points.append(((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
    if points and page.rotation:
        matrix = page.rotation_matrix
        points = [tuple(fitz.Point(x, y) * matrix) for x, y in points]
    return points


def detected_lines(page, engine):
    dpi = _dpi_for(page, GATE_DPI)
    boxes = getattr(engine(render(page, dpi), **DETECT_ONLY), "boxes", None)
    scale = 72.0 / dpi
    lines = []
    for box in ([] if boxes is None else boxes):
        xs = [float(point[0]) * scale for point in box]
        ys = [float(point[1]) * scale for point in box]
        lines.append((min(xs), min(ys), max(xs), max(ys)))
    return lines


def page_has_text_it_does_not_hold(page, engine):
    lines = detected_lines(page, engine)
    if not lines:
        return False
    points = visible_character_centres(page)
    if not points:
        return True
    centres = np.asarray(points)
    for x0, y0, x1, y1 in lines:
        inside = (centres[:, 0] >= x0) & (centres[:, 0] <= x1) & (centres[:, 1] >= y0) & (centres[:, 1] <= y1)
        if not inside.any():
            return True
    return False


def page_requires_ocr(page, check, engine=None, own=False):
    if text_is_garbled(page.get_text()):
        return True
    if own:
        return False
    if check == "quick":
        return page_needs_ocr(page)
    return page_has_text_it_does_not_hold(page, engine)


def _results(result):
    if result.txts is None or result.boxes is None:
        return [], [], []
    texts, boxes = list(result.txts), list(result.boxes)
    scores = list(result.scores) if result.scores is not None else []
    scores = scores + [1.0] * (len(texts) - len(scores))
    return texts, boxes, scores


def _text_mass(texts, scores):
    return float(sum(score * len(text) for text, score in zip(texts, scores) if len(text) >= 3))


def _mean_confidence(texts, scores):
    kept = [score for text, score in zip(texts, scores) if len(text) >= 3]
    return float(sum(kept) / len(kept)) if kept else 0.0


def _horizontal_fraction(texts, boxes):
    horizontal = total = 0
    for text, box in zip(texts, boxes):
        if len(text) < 3:
            continue
        points = np.asarray(box, dtype=float).reshape(-1, 2)
        total += len(text)
        if points[:, 0].max() - points[:, 0].min() >= points[:, 1].max() - points[:, 1].min():
            horizontal += len(text)
    return horizontal / total if total else 0.0


def _orientation_score(texts, boxes, scores):
    return _text_mass(texts, scores) * _horizontal_fraction(texts, boxes)


def _read_upright(image, engine):
    first = _results(engine(image, text_score=TEXT_SCORE, **READ))
    if _mean_confidence(first[0], first[2]) >= ORIENT_ACCEPT_CONF and _horizontal_fraction(first[0], first[1]) >= 0.5:
        return first
    base = _orientation_score(*first)
    best = None
    for variant in (np.rot90(image, 1), np.rot90(image, 2), np.rot90(image, 3), 255 - image):
        variant = np.ascontiguousarray(variant)
        probe = _results(engine(np.ascontiguousarray(variant[::2, ::2]), text_score=TEXT_SCORE, **READ))
        score = _orientation_score(*probe)
        if best is None or score > best[0]:
            best = (score, variant)
    turned = _results(engine(best[1], text_score=TEXT_SCORE, **READ))
    return turned if _orientation_score(*turned) > base * ORIENT_MARGIN else first


def read_page_rapidocr(page, engine):
    texts, boxes, _ = _read_upright(render(page, RAPIDOCR_DPI), engine)
    lines = [(text, box) for text, box in zip(texts, boxes) if text and str(text).strip()]
    if not lines:
        return ""
    try:
        order = column_reading_order([box for _, box in lines])
    except Exception:
        order = None
    if order is not None:
        lines = [lines[i] for i in order]
    return "\n".join(str(text) for text, _ in lines)


def read_page_tesseract(page, engine):
    from PIL import Image
    dpi = _dpi_for(page, TESSERACT_DPI)
    pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    reader, orientation = engine
    if orientation is not None:
        try:
            orientation.SetImage(image)
            found = orientation.DetectOrientationScript()
        except Exception:
            found = None
        if found and found.get("orient_deg") and found.get("orient_conf", 0) >= TESSERACT_ORIENT_CONF:
            image = image.rotate(found["orient_deg"], expand=True)
    reader.SetImage(image)
    reader.SetSourceResolution(dpi)
    return reader.GetUTF8Text().strip()


_worker = {}


def start_worker(check, reader, threads):
    _worker.clear()
    if check == "detector":
        _worker["detector"] = rapidocr_engine("small", threads)
    if reader == "rapidocr":
        _worker["reader"] = rapidocr_engine("medium", threads)
    elif reader == "tesseract":
        _worker["reader"] = tesseract_engine()


def check_pages(path, pages, check):
    flagged = []
    with fitz.open(path) as document:
        own = produced_by_ocr(document)
        for number in pages:
            try:
                if page_requires_ocr(document[number - 1], check, _worker.get("detector"), own):
                    flagged.append(number)
            except Exception as error:
                logger.error(f"Could not check page {number} of {Path(path).name}: {error}")
    return path, pages, flagged


def read_pages(path, pages, reader):
    texts = {}
    read = read_page_tesseract if reader == "tesseract" else read_page_rapidocr
    with fitz.open(path) as document:
        for number in pages:
            try:
                texts[number] = read(document[number - 1], _worker["reader"])
            except Exception as error:
                logger.error(f"Could not read page {number} of {Path(path).name}: {error}")
    return path, pages, texts
