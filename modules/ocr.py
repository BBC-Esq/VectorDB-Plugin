import os
import io
import tempfile
import threading
import queue
import time
from pathlib import Path
from io import BytesIO
from abc import ABC, abstractmethod
from core.constants import PROJECT_ROOT
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue

import fitz
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
        output_pdf = fitz.open()
        for temp_pdf_path, _ in results:
            output_pdf.insert_pdf(fitz.open(temp_pdf_path))
            Path(temp_pdf_path).unlink(missing_ok=True)
        output_pdf.save(output_path)
        output_pdf.close()
        self.optimize_final_pdf(pdf_path, output_path)
        self.cleanup_temp_pdfs()
        if self.progress_queue:
            self.progress_queue.put(('done', None))

    def process_page(self, page_num: int, pdf_path: str) -> Tuple[int, str]:
        fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf", dir=self.temp_dir)
        os.close(fd)
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_num]
        out_pdf = fitz.open()
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
        out_pdf.close()
        pdf_document.close()
        return page_num, temp_pdf_path

    def optimize_final_pdf(self, original_pdf_path: Path, ocr_pdf_path: Path) -> None:
        with fitz.open(original_pdf_path) as original_doc:
            orig_pages = []
            for page in original_doc:
                orig_pages.append({'width': page.rect.width, 'height': page.rect.height, 'mediabox': page.mediabox, 'cropbox': getattr(page, 'cropbox', None)})
        temp_path = str(ocr_pdf_path) + ".optimized"
        with fitz.open(ocr_pdf_path) as ocr_doc:
            for i, page in enumerate(ocr_doc):
                if i < len(orig_pages):
                    orig = orig_pages[i]
                    page.set_mediabox(orig['mediabox'])
                    if orig['cropbox']:
                        try:
                            cropbox = orig['cropbox']
                            mediabox = orig['mediabox']
                            if cropbox[0] >= mediabox[0] and cropbox[1] >= mediabox[1] and cropbox[2] <= mediabox[2] and cropbox[3] <= mediabox[3]:
                                page.set_cropbox(cropbox)
                        except ValueError:
                            pass
            ocr_doc.save(temp_path, garbage=4, deflate=True, clean=True, linear=True)
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
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    processor.initialize()
    try:
        for pdf_path in pdf_paths:
            output_path = None
            if output_dir:
                output_path = output_dir / f"{pdf_path.stem}_ocr.txt"
            processor.process_document(pdf_path, output_path)
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
    try:
        while True:
            try:
                msg = progress_queue.get(timeout=1.0)
                cmd, data = msg
                if cmd == 'total':
                    total_pages = data
                    pbar = tqdm.tqdm(total=total_pages, desc="Processing pages")
                elif cmd == 'update':
                    if pbar:
                        pbar.update(data)
                elif cmd == 'done':
                    break
            except queue.Empty:
                if not process.is_alive():
                    break
    finally:
        if pbar:
            pbar.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=3.0)
            if process.is_alive():
                process.kill()
        time.sleep(0.5)
