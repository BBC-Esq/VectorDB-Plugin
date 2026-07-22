import os
import io
import html
import tempfile
import threading
import queue
import time
from pathlib import Path
from io import BytesIO
from abc import ABC, abstractmethod
from core.constants import PROJECT_ROOT
from core.pdf_ocr_gate import page_needs_ocr
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue

import fitz
import numpy as np
import psutil
from PIL import Image
import tesserocr
from ocrmypdf.hocrtransform import HocrTransform
import tqdm
from typing import Union, List, Tuple

thread_local = threading.local()

class OCRProcessor(ABC):
    def __init__(self, zoom: int = 2, progress_queue: Queue = None):
        self.zoom = zoom
        self.show_progress = False
        self.progress_queue = progress_queue
        backend_name = self.__class__.__name__
        print(f"\033[92mUsing {backend_name} backend\033[0m")
        if backend_name == "TesseractOCR":
            thread_count = self.get_optimal_threads()
            print(f"\033[92mUsing up to {thread_count} threads\033[0m")

    def convert_page_to_image(self, page) -> Image.Image:
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))

    @abstractmethod
    def process_page(self, page_num: int, pdf_path: str) -> Tuple[int, str]:
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def clean_text(self, text: str) -> str:
        pass

    def validate_pdf(self, pdf_path: Path) -> bool:
        try:
            with fitz.open(str(pdf_path)) as doc:
                if doc.page_count == 0:
                    return False
                _ = doc[0].get_text()
            return True
        except Exception:
            return False

    def process_document(self, pdf_path: Path, output_path: Path = None):
        if not self.validate_pdf(pdf_path):
            raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}")
        if output_path is None:
            output_path = pdf_path.with_suffix('.txt')
        with fitz.open(str(pdf_path)) as pdf_document:
            total_pages = len(pdf_document)
        if self.progress_queue:
            self.progress_queue.put(('total', total_pages))
        results = {}
        with ThreadPoolExecutor(max_workers=self.get_optimal_threads()) as executor:
            future_to_page = {
                executor.submit(self.process_page, page_num, str(pdf_path)): page_num
                for page_num in range(total_pages)
            }
            for future in as_completed(future_to_page):
                page_num, processed_text = future.result()
                results[page_num] = processed_text
                if self.progress_queue:
                    self.progress_queue.put(('update', 1))
        with open(output_path, 'w', encoding='utf-8') as f:
            for page_num in range(total_pages):
                text = results.get(page_num, '').strip()
                if text:
                    f.write(f"[[page{page_num + 1}]]{text}")
        if self.progress_queue:
            self.progress_queue.put(('done', None))

    @staticmethod
    def get_optimal_threads() -> int:
        return max(4, psutil.cpu_count(logical=True) - 3)

class TesseractOCR(OCRProcessor):
    def __init__(self, zoom: int = 2, progress_queue: Queue = None):
        super().__init__(zoom, progress_queue)
        self.tessdata_path = None
        self.temp_dir = None
        self.show_progress = True

    def initialize(self):
        script_dir = PROJECT_ROOT
        self.temp_dir = script_dir / "temp_ocr"
        self.temp_dir.mkdir(exist_ok=True)
        os.environ['TMP'] = str(self.temp_dir)
        os.environ['TEMP'] = str(self.temp_dir)
        tempfile.tempdir = str(self.temp_dir)
        self.tessdata_path = script_dir / 'share' / 'tessdata'
        os.environ['TESSDATA_PREFIX'] = str(self.tessdata_path)

    def clean_text(self, text: str) -> str:
        return text

    def cleanup(self):
        self.cleanup_temp_pdfs()
        if 'TESSDATA_PREFIX' in os.environ:
            del os.environ['TESSDATA_PREFIX']

    def process_document(self, pdf_path: Path, output_path: Path = None):
        if not self.validate_pdf(pdf_path):
            raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}")
        if output_path is None:
            output_path = pdf_path.with_stem(f"{pdf_path.stem}_OCR")
        if self.temp_dir is None:
            self.initialize()
        self.cleanup_temp_pdfs()
        with fitz.open(str(pdf_path)) as pdf_document:
            num_pages = len(pdf_document)
        if self.progress_queue:
            self.progress_queue.put(('total', num_pages))
        results = []
        with ThreadPoolExecutor(max_workers=self.get_optimal_threads()) as executor:
            futures = {executor.submit(self.process_page, page_num, str(pdf_path)): page_num for page_num in range(num_pages)}
            for future in as_completed(futures):
                page_num, temp_pdf_path = future.result()
                results.append((temp_pdf_path, page_num))
                if self.progress_queue:
                    self.progress_queue.put(('update', 1))
        results.sort(key=lambda x: x[1])
        with fitz.open() as output_pdf:
            for temp_pdf_path, _ in results:
                with fitz.open(temp_pdf_path) as src:
                    output_pdf.insert_pdf(src)
                Path(temp_pdf_path).unlink(missing_ok=True)
            output_pdf.save(output_path)
        self.optimize_final_pdf(pdf_path, output_path)
        self.cleanup_temp_pdfs()
        if self.progress_queue:
            self.progress_queue.put(('done', None))

    def process_page(self, page_num: int, pdf_path: str) -> Tuple[int, str]:
        fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf", dir=self.temp_dir)
        os.close(fd)
        with fitz.open(pdf_path) as pdf_document, fitz.open() as out_pdf:
            page = pdf_document[page_num]
            api = getattr(thread_local, 'api', None)
            if api is None:
                api = tesserocr.PyTessBaseAPI(lang="eng", path=str(self.tessdata_path))
                thread_local.api = api
            page.remove_rotation()
            pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
            pil_image = Image.open(BytesIO(pix.tobytes("png")))
            api.SetImage(pil_image)
            hocr_text = api.GetHOCRText(0)
            hocr_output = f"{self.temp_dir}/page_{page_num}.hocr"
            Path(hocr_output).write_text(hocr_text, encoding="utf-8")
            fd, text_pdf = tempfile.mkstemp(suffix=".pdf", dir=self.temp_dir)
            os.close(fd)
            pdf_width_pts = page.rect.width
            pdf_height_pts = page.rect.height
            dpi_x = (pix.width * 72) / pdf_width_pts
            dpi_y = (pix.height * 72) / pdf_height_pts
            dpi = (dpi_x + dpi_y) / 2.0
            hocr_transform = HocrTransform(hocr_filename=hocr_output, dpi=dpi)
            # HocrTransform.to_pdf reads self.width/self.height. __init__ tries to set
            # them from the hOCR <div class="ocr_page"> coords, but tesserocr's hOCR
            # may omit that div (or its bbox), leaving the attrs undefined and causing
            # AttributeError. Force them to the true PDF page dimensions in pts.
            hocr_transform.width = pdf_width_pts
            hocr_transform.height = pdf_height_pts
            hocr_transform.to_pdf(out_filename=text_pdf, invisible_text=True)
            out_pdf.insert_pdf(page.parent, from_page=page_num, to_page=page_num)
            with fitz.open(text_pdf) as text_page:
                out_pdf[0].show_pdf_page(out_pdf[0].rect, text_page, 0, overlay=True)
            Path(hocr_output).unlink(missing_ok=True)
            for _ in range(10):
                try:
                    Path(text_pdf).unlink()
                    break
                except PermissionError:
                    time.sleep(0.1)
            out_pdf.save(temp_pdf_path)
        return page_num, temp_pdf_path

    def optimize_final_pdf(self, original_pdf_path: Path, ocr_pdf_path: Path) -> None:
        with fitz.open(original_pdf_path) as original_doc:
            orig_pages = []
            for page in original_doc:
                orig_pages.append({'width': page.rect.width, 'height': page.rect.height, 'rotation': page.rotation, 'mediabox': page.mediabox, 'cropbox': getattr(page, 'cropbox', None)})
        temp_path = str(ocr_pdf_path) + ".optimized"
        with fitz.open(ocr_pdf_path) as ocr_doc:
            for i, page in enumerate(ocr_doc):
                if i < len(orig_pages):
                    orig = orig_pages[i]
                    if orig['rotation'] in (90, 270):
                        page.set_mediabox(fitz.Rect(0, 0, orig['width'], orig['height']))
                        continue
                    page.set_mediabox(orig['mediabox'])
                    if orig['cropbox']:
                        try:
                            cropbox = orig['cropbox']
                            mediabox = orig['mediabox']
                            if cropbox[0] >= mediabox[0] and cropbox[1] >= mediabox[1] and cropbox[2] <= mediabox[2] and cropbox[3] <= mediabox[3]:
                                page.set_cropbox(cropbox)
                        except ValueError:
                            pass
            ocr_doc.save(temp_path, garbage=4, deflate=True, clean=True)
        os.replace(temp_path, ocr_pdf_path)

    def cleanup_temp_pdfs(self):
        if self.temp_dir is None:
            return
        for temp_file in Path(self.temp_dir).glob("tmp*.pdf"):
            try:
                temp_file.unlink()
            except PermissionError:
                pass

class RapidOCRBackend(OCRProcessor):
    _MODELS_SUBDIR = PROJECT_ROOT / "models" / "ocr" / "rapidocr"
    _MODEL_SOURCES = {
        "PP-OCRv6_det_small.onnx": ("PaddlePaddle/PP-OCRv6_small_det_onnx", "inference.onnx"),
        "PP-OCRv6_rec_small.onnx": ("PaddlePaddle/PP-OCRv6_small_rec_onnx", "inference.onnx"),
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx": ("SWHL/RapidOCR",
                                                 "PP-OCRv1/ch_ppocr_mobile_v2.0_cls_infer.onnx"),
    }

    def __init__(self, zoom: int = 2, progress_queue: Queue = None):
        super().__init__(zoom, progress_queue)
        self.temp_dir = None
        self.engine = None
        self.show_progress = True

    def _ensure_model(self, name: str) -> Path:
        p = self._MODELS_SUBDIR / name
        if p.exists():
            return p
        self._MODELS_SUBDIR.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download
        if name == "PP-OCRv6_rec_keys.txt":
            import yaml
            yml = hf_hub_download("PaddlePaddle/PP-OCRv6_small_rec_onnx", "inference.yml")
            chars = yaml.safe_load(Path(yml).read_text(encoding="utf-8"))["PostProcess"]["character_dict"]
            p.write_bytes("\n".join(chars).encode("utf-8"))
            return p
        repo, fn = self._MODEL_SOURCES[name]
        print(f"\033[92m[RapidOCR] fetching {name} from HuggingFace {repo}\033[0m")
        src = hf_hub_download(repo, fn)
        p.write_bytes(Path(src).read_bytes())
        return p

    def initialize(self):
        self.temp_dir = PROJECT_ROOT / "temp_ocr"
        self.temp_dir.mkdir(exist_ok=True)
        os.environ['TMP'] = str(self.temp_dir)
        os.environ['TEMP'] = str(self.temp_dir)
        tempfile.tempdir = str(self.temp_dir)
        from rapidocr import RapidOCR, OCRVersion
        det = self._ensure_model("PP-OCRv6_det_small.onnx")
        rec = self._ensure_model("PP-OCRv6_rec_small.onnx")
        keys = self._ensure_model("PP-OCRv6_rec_keys.txt")
        cls = self._ensure_model("ch_ppocr_mobile_v2.0_cls_mobile.onnx")
        threads = max(4, psutil.cpu_count(logical=True) - 4)
        self.engine = RapidOCR(params={
            "Det.model_path": str(det), "Det.ocr_version": OCRVersion.PPOCRV6,
            "Rec.model_path": str(rec), "Rec.rec_keys_path": str(keys),
            "Cls.model_path": str(cls),
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        })

    def clean_text(self, text: str) -> str:
        return text

    def cleanup(self):
        self.cleanup_temp_pdfs()

    @staticmethod
    def _poly_to_bbox(poly) -> Tuple[int, int, int, int]:
        a = np.asarray(poly, dtype=float)
        return int(a[:, 0].min()), int(a[:, 1].min()), int(a[:, 0].max()), int(a[:, 1].max())

    @classmethod
    def _build_hocr(cls, txts, boxes, scores, width: int, height: int):
        spans = []
        emitted = dropped = flagged = 0
        confs = []
        for i, (t, poly, sc) in enumerate(zip(txts, boxes, scores)):
            if not t:
                continue
            x0, y0, x1, y1 = cls._poly_to_bbox(poly)
            if x1 - x0 <= 1 or y1 - y0 <= 1:
                continue
            if sc < cls._REC_HARD_FLOOR:
                dropped += 1
                continue
            if sc < cls._REC_FLAG_FLOOR:
                flagged += 1
            confs.append(sc)
            wconf = max(0, min(100, int(round(sc * 100))))
            esc = html.escape(str(t))
            spans.append(
                f"<span class='ocr_line' id='line_{i}' title='bbox {x0} {y0} {x1} {y1}'>"
                f"<span class='ocrx_word' id='word_{i}' title='bbox {x0} {y0} {x1} {y1}; x_wconf {wconf}'>"
                f"{esc}</span></span>")
            emitted += 1
        body = "\n".join(spans)
        hocr = (
            "<?xml version='1.0' encoding='UTF-8'?>\n"
            "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" "
            "\"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n"
            "<html xmlns='http://www.w3.org/1999/xhtml'><head>"
            "<meta http-equiv='Content-Type' content='text/html;charset=utf-8'/>"
            "<meta name='ocr-system' content='rapidocr'/>"
            "<meta name='ocr-capabilities' content='ocr_page ocr_line ocrx_word'/></head><body>"
            f"<div class='ocr_page' id='page_1' title='bbox 0 0 {width} {height}'>"
            f"<div class='ocr_carea' title='bbox 0 0 {width} {height}'><p class='ocr_par'>"
            f"{body}</p></div></div></body></html>")
        mean_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return hocr, {'emitted': emitted, 'dropped': dropped, 'flagged': flagged, 'mean_conf': mean_conf}

    _ORIENT_ACCEPT_CONF = 0.90
    _ORIENT_MARGIN = 1.20

    _REC_HARD_FLOOR = 0.30
    _REC_FLAG_FLOOR = 0.60
    _PAGE_LOWCONF = 0.75
    _MAX_PIXELS = 30_000_000
    _MAX_DIM = 12000

    def _signal(self, kind: str, msg: str, data: dict = None):
        print(f"\033[93m[RapidOCR] {kind}: {msg}\033[0m", flush=True)
        if self.progress_queue:
            try:
                payload = {'msg': msg}
                if data:
                    payload.update(data)
                self.progress_queue.put((kind, payload))
            except Exception:
                pass

    def _safe_zoom(self, page) -> float:
        wpt = max(float(page.rect.width), 1.0)
        hpt = max(float(page.rect.height), 1.0)
        z = float(self.zoom)
        z = min(z, self._MAX_DIM / max(wpt, hpt), (self._MAX_PIXELS / (wpt * hpt)) ** 0.5)
        return max(z, 0.1)

    @staticmethod
    def _unpack(res):
        txts = list(res.txts) if res.txts is not None else []
        boxes = list(res.boxes) if res.boxes is not None else []
        scores = list(res.scores) if res.scores is not None else []
        if len(scores) < len(txts):
            scores = scores + [1.0] * (len(txts) - len(scores))
        return txts, boxes, scores

    @staticmethod
    def _text_mass(res) -> float:
        if res.txts is None or res.scores is None:
            return 0.0
        return float(sum(s * len(t) for t, s in zip(res.txts, res.scores) if len(t) >= 3))

    @staticmethod
    def _mean_conf(res) -> float:
        if res.txts is None or res.scores is None:
            return 0.0
        subs = [s for t, s in zip(res.txts, res.scores) if len(t) >= 3]
        return float(sum(subs) / len(subs)) if subs else 0.0

    @staticmethod
    def _remap_k1(poly, w, h):
        return np.array([[w - 1 - p[1], p[0]] for p in poly])

    @staticmethod
    def _remap_k2(poly, w, h):
        return np.array([[w - 1 - p[0], h - 1 - p[1]] for p in poly])

    @staticmethod
    def _remap_k3(poly, w, h):
        return np.array([[p[1], h - 1 - p[0]] for p in poly])

    def _ocr_oriented(self, img, w: int, h: int):
        r0 = self.engine(img, use_cls=False)
        if self._mean_conf(r0) >= self._ORIENT_ACCEPT_CONF:
            return self._unpack(r0) + ('none',)
        base_mass = self._text_mass(r0)
        variants = (
            ('rot90', np.ascontiguousarray(np.rot90(img, 1)), self._remap_k1),
            ('rot180', np.ascontiguousarray(np.rot90(img, 2)), self._remap_k2),
            ('rot270', np.ascontiguousarray(np.rot90(img, 3)), self._remap_k3),
            ('invert', np.ascontiguousarray(255 - img), None),
        )
        ranked = []
        for name, vimg, remap in variants:
            probe = np.ascontiguousarray(vimg[::2, ::2])
            ranked.append((self._text_mass(self.engine(probe, use_cls=False)), name, vimg, remap))
        ranked.sort(key=lambda s: s[0], reverse=True)
        _, name, vimg, remap = ranked[0]
        r_win = self.engine(vimg, use_cls=False)
        if self._text_mass(r_win) > base_mass * self._ORIENT_MARGIN:
            txts, boxes, scores = self._unpack(r_win)
            if remap is not None:
                boxes = [remap(np.asarray(p), w, h) for p in boxes]
            return txts, boxes, scores, name
        return self._unpack(r0) + ('none',)

    def process_page(self, page_num: int, pdf_path: str) -> Tuple[int, str]:
        fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf", dir=self.temp_dir)
        os.close(fd)
        with fitz.open(pdf_path) as pdf_document, fitz.open() as out_pdf:
            page = pdf_document[page_num]
            page.remove_rotation()
            out_pdf.insert_pdf(page.parent, from_page=page_num, to_page=page_num)
            try:
                needs = page_needs_ocr(page)
            except Exception:
                needs = True
            if not needs:
                out_pdf.save(temp_pdf_path)
                return page_num, temp_pdf_path
            try:
                z = self._safe_zoom(page)
                pix = page.get_pixmap(matrix=fitz.Matrix(z, z))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)[:, :, :3][:, :, ::-1].copy()
                txts, boxes, scores, orient = self._ocr_oriented(img, pix.width, pix.height)
                if orient != 'none':
                    self._signal('oriented',
                                 f"page {page_num + 1}: auto-corrected orientation ({orient})",
                                 {'page': page_num + 1, 'orient': orient})
                if txts and boxes:
                    hocr_text, stats = self._build_hocr(txts, boxes, scores, pix.width, pix.height)
                    if stats['emitted'] > 0:
                        hocr_output = f"{self.temp_dir}/page_{page_num}.hocr"
                        Path(hocr_output).write_text(hocr_text, encoding="utf-8")
                        fd, text_pdf = tempfile.mkstemp(suffix=".pdf", dir=self.temp_dir)
                        os.close(fd)
                        pdf_width_pts = page.rect.width
                        pdf_height_pts = page.rect.height
                        dpi = ((pix.width * 72) / pdf_width_pts + (pix.height * 72) / pdf_height_pts) / 2.0
                        hocr_transform = HocrTransform(hocr_filename=hocr_output, dpi=dpi)
                        hocr_transform.width = pdf_width_pts
                        hocr_transform.height = pdf_height_pts
                        hocr_transform.to_pdf(out_filename=text_pdf, invisible_text=True)
                        with fitz.open(text_pdf) as text_page:
                            out_pdf[0].show_pdf_page(out_pdf[0].rect, text_page, 0, overlay=True)
                        Path(hocr_output).unlink(missing_ok=True)
                        for _ in range(10):
                            try:
                                Path(text_pdf).unlink()
                                break
                            except PermissionError:
                                time.sleep(0.1)
                    if stats['dropped'] or stats['flagged'] or stats['mean_conf'] < self._PAGE_LOWCONF:
                        self._signal(
                            'lowconf',
                            f"page {page_num + 1}: mean_conf={stats['mean_conf']:.2f} "
                            f"flagged={stats['flagged']} dropped={stats['dropped']} "
                            f"(consider a vision re-read)",
                            {'page': page_num + 1, **stats})
                else:
                    ink = float((img.min(axis=2) < 100).mean())
                    self._signal(
                        'notext',
                        f"page {page_num + 1}: no text detected "
                        f"(ink={ink * 100:.1f}%; "
                        f"{'looks blank' if ink < 0.002 else 'has content but OCR found nothing'})",
                        {'page': page_num + 1, 'ink_frac': ink})
            except Exception as e:
                self._signal(
                    'pageerror',
                    f"page {page_num + 1} failed ({type(e).__name__}: {e}); kept the page image, no text layer",
                    {'page': page_num + 1, 'error': f"{type(e).__name__}: {e}"})
            out_pdf.save(temp_pdf_path)
        return page_num, temp_pdf_path

    def process_document(self, pdf_path: Path, output_path: Path = None):
        src_had_text = False
        try:
            with fitz.open(str(pdf_path)) as _doc:
                if _doc.needs_pass and not _doc.authenticate(''):
                    raise ValueError(f"Encrypted PDF requires a password (cannot OCR): {pdf_path.name}")
                src_had_text = any(pg.get_text().strip() for pg in _doc)
        except ValueError:
            raise
        except Exception:
            pass
        if not self.validate_pdf(pdf_path):
            raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}")
        if output_path is None:
            output_path = pdf_path.with_stem(f"{pdf_path.stem}_OCR")
        if self.temp_dir is None or self.engine is None:
            self.initialize()
        self.cleanup_temp_pdfs()
        with fitz.open(str(pdf_path)) as pdf_document:
            num_pages = len(pdf_document)
        if self.progress_queue:
            self.progress_queue.put(('total', num_pages))
        results = []
        for page_num in range(num_pages):
            _, temp_pdf_path = self.process_page(page_num, str(pdf_path))
            results.append((temp_pdf_path, page_num))
            if self.progress_queue:
                self.progress_queue.put(('update', 1))
        results.sort(key=lambda x: x[1])
        with fitz.open() as output_pdf:
            for temp_pdf_path, _ in results:
                with fitz.open(temp_pdf_path) as src:
                    output_pdf.insert_pdf(src)
                Path(temp_pdf_path).unlink(missing_ok=True)
            output_pdf.save(output_path)
        self.optimize_final_pdf(pdf_path, output_path)
        self.cleanup_temp_pdfs()
        self._verify_output(pdf_path, output_path, num_pages, src_had_text)
        if self.progress_queue:
            self.progress_queue.put(('done', None))

    def _verify_output(self, pdf_path: Path, output_path: Path, num_pages: int, src_had_text: bool):
        try:
            if not Path(output_path).exists():
                self._signal('verifyfail', f"output not written: {Path(output_path).name}")
                return
            with fitz.open(str(output_path)) as out:
                if len(out) != num_pages:
                    self._signal('verifyfail',
                                 f"page count mismatch in {Path(output_path).name}: in={num_pages} out={len(out)}")
                if not src_had_text:
                    if not "".join(pg.get_text() for pg in out).strip():
                        self._signal('verifyfail',
                                     f"image-only input produced NO extractable text: {pdf_path.name}")
        except Exception as e:
            self._signal('verifyfail', f"could not verify {Path(output_path).name}: {type(e).__name__}: {e}")

    def optimize_final_pdf(self, original_pdf_path: Path, ocr_pdf_path: Path) -> None:
        with fitz.open(original_pdf_path) as original_doc:
            orig_pages = []
            for page in original_doc:
                orig_pages.append({'width': page.rect.width, 'height': page.rect.height,
                                   'rotation': page.rotation, 'mediabox': page.mediabox,
                                   'cropbox': getattr(page, 'cropbox', None)})
            src_meta = dict(original_doc.metadata or {})
            try:
                src_lang = original_doc.xref_get_key(original_doc.pdf_catalog(), "Lang")
            except Exception:
                src_lang = (None, None)
        temp_path = str(ocr_pdf_path) + ".optimized"
        with fitz.open(ocr_pdf_path) as ocr_doc:
            for i, page in enumerate(ocr_doc):
                if i < len(orig_pages):
                    orig = orig_pages[i]
                    if orig['rotation'] in (90, 270):
                        page.set_mediabox(fitz.Rect(0, 0, orig['width'], orig['height']))
                    else:
                        page.set_mediabox(orig['mediabox'])
                        if orig['cropbox']:
                            try:
                                cropbox = orig['cropbox']
                                mediabox = orig['mediabox']
                                if cropbox[0] >= mediabox[0] and cropbox[1] >= mediabox[1] and cropbox[2] <= mediabox[2] and cropbox[3] <= mediabox[3]:
                                    page.set_cropbox(cropbox)
                            except ValueError:
                                pass
            meta = {k: v for k, v in src_meta.items()
                    if v and k in ('title', 'author', 'subject', 'keywords', 'creator',
                                   'creationDate', 'trapped')}
            meta['producer'] = 'VectorDB-Plugin RapidOCR (PP-OCRv6)'
            meta['modDate'] = fitz.get_pdf_now()
            try:
                ocr_doc.set_metadata(meta)
            except Exception:
                pass
            try:
                if src_lang[0] == 'string' and src_lang[1]:
                    ocr_doc.xref_set_key(ocr_doc.pdf_catalog(), "Lang", fitz.get_pdf_str(src_lang[1]))
            except Exception:
                pass
            ocr_doc.save(temp_path, garbage=4, deflate=True, clean=True)
        os.replace(temp_path, ocr_pdf_path)

    def cleanup_temp_pdfs(self):
        if self.temp_dir is None:
            return
        for temp_file in Path(self.temp_dir).glob("tmp*.pdf"):
            try:
                temp_file.unlink()
            except PermissionError:
                pass


def _process_documents_worker(pdf_paths: List[Path], backend: str, model_path: str, output_dir: Path, progress_queue: Queue):
    if backend.lower() == 'tesseract':
        processor = TesseractOCR(progress_queue=progress_queue)
    elif backend.lower() == 'rapidocr':
        processor = RapidOCRBackend(progress_queue=progress_queue)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    processor.initialize()
    try:
        for pdf_path in pdf_paths:
            output_path = None
            if output_dir:
                output_path = output_dir / f"{pdf_path.stem}_ocr.txt"
            try:
                processor.process_document(pdf_path, output_path)
            except Exception as e:
                print(f"\033[93m[OCR] file failed: {pdf_path.name} ({type(e).__name__}: {e})\033[0m", flush=True)
                try:
                    progress_queue.put(('fileerror', {'file': str(pdf_path), 'error': f"{type(e).__name__}: {e}"}))
                except Exception:
                    pass
    finally:
        if hasattr(processor, 'cleanup'):
            processor.cleanup()

def process_documents(pdf_paths: Union[Path, List[Path]], backend: str = 'tesseract', model_path: str = None, output_dir: Path = None):
    if isinstance(pdf_paths, Path):
        pdf_paths = [pdf_paths]
    progress_queue = Queue()
    process = Process(target=_process_documents_worker, args=(pdf_paths, backend, model_path, output_dir, progress_queue))
    process.start()
    total_pages = None
    pbar = None
    documents_done = 0
    events = {'lowconf': [], 'notext': [], 'oriented': [], 'pageerror': [], 'verifyfail': [], 'fileerror': []}
    try:
        while True:
            try:
                msg = progress_queue.get(timeout=1.0)
                cmd, data = msg
                if cmd == 'total':
                    total_pages = data
                    if pbar:
                        pbar.close()
                    pbar = tqdm.tqdm(total=total_pages, desc="Processing pages")
                elif cmd == 'update':
                    if pbar:
                        pbar.update(data)
                elif cmd == 'done':
                    documents_done += 1
                    if documents_done + len(events['fileerror']) >= len(pdf_paths):
                        break
                elif cmd in events:
                    events[cmd].append(data)
                    if cmd == 'fileerror' and documents_done + len(events['fileerror']) >= len(pdf_paths):
                        break
            except queue.Empty:
                if not process.is_alive():
                    break
    finally:
        if pbar:
            pbar.close()
        if process.is_alive():
            process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        time.sleep(0.5)

    if process.exitcode is not None and process.exitcode != 0:
        raise RuntimeError(f"OCR worker exited with code {process.exitcode}")

    if events['fileerror'] and len(events['fileerror']) >= len(pdf_paths):
        details = "; ".join(f"{Path(e['file']).name}: {e['error']}" for e in events['fileerror'])
        raise RuntimeError(f"OCR failed for all {len(pdf_paths)} file(s): {details}")

    n_low, n_nt, n_or, n_err, n_bad, n_file = (len(events['lowconf']), len(events['notext']),
                                               len(events['oriented']), len(events['pageerror']),
                                               len(events['verifyfail']), len(events['fileerror']))
    if n_low or n_nt or n_or or n_err or n_bad or n_file:
        print(f"\033[93m[OCR] summary: {n_low} low-confidence page(s), "
              f"{n_nt} no-text page(s), {n_or} auto-oriented page(s), "
              f"{n_err} page error(s), {n_bad} verification warning(s), "
              f"{n_file} failed file(s)\033[0m", flush=True)
    return {'lowconf': events['lowconf'], 'notext': events['notext'], 'oriented': events['oriented'],
            'pageerror': events['pageerror'], 'verifyfail': events['verifyfail'],
            'fileerror': events['fileerror']}
