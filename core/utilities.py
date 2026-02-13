import importlib
import importlib.metadata
import importlib.util
import os
import threading
import logging
import platform
import shutil
import sys
from pathlib import Path
import psutil
import subprocess
import re

import torch
import yaml
from packaging import version
from PySide6.QtCore import QRunnable, QObject, Signal, QThreadPool
from PySide6.QtWidgets import QApplication, QMessageBox
from termcolor import cprint

from core.constants import PROJECT_ROOT

def set_cuda_paths():
    import sys
    import os
    from pathlib import Path
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path_runtime = nvidia_base_path / 'cuda_runtime' / 'bin'
    cuda_path_runtime_lib = nvidia_base_path / 'cuda_runtime' / 'lib' / 'x64'
    cuda_path_runtime_include = nvidia_base_path / 'cuda_runtime' / 'include'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    nvrtc_path = nvidia_base_path / 'cuda_nvrtc' / 'bin'
    nvcc_path = nvidia_base_path / 'cuda_nvcc' / 'bin'
    paths_to_add = [
        str(cuda_path_runtime),
        str(cuda_path_runtime_lib),
        str(cuda_path_runtime_include),
        str(cublas_path),
        str(cudnn_path),
        str(nvrtc_path),
        str(nvcc_path),
    ]
    current_value = os.environ.get('PATH', '')
    new_value = os.pathsep.join(paths_to_add + ([current_value] if current_value else []))
    os.environ['PATH'] = new_value

    triton_cuda_path = nvidia_base_path / 'cuda_runtime'
    current_cuda_path = os.environ.get('CUDA_PATH', '')
    new_cuda_path = os.pathsep.join([str(triton_cuda_path)] + ([current_cuda_path] if current_cuda_path else []))
    os.environ['CUDA_PATH'] = new_cuda_path


def check_backend_dependencies(backend_name: str, interactive: bool = True) -> bool:
    from core.constants import BACKEND_DEPENDENCIES

    required_packages = BACKEND_DEPENDENCIES.get(backend_name, {})
    
    if not required_packages:
        return True

    return check_and_install_dependencies(
        required_packages,
        backend_name=backend_name.title(),
        interactive=interactive
    )

def is_package_available(pkg_name: str) -> tuple[bool, str]:
    import importlib.util
    import importlib.metadata
    
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    return package_exists, package_version

def verify_installation(package_name: str, expected_version: str) -> bool:
    try:
        import importlib.metadata
        installed_version = importlib.metadata.version(package_name)
        return installed_version == expected_version
    except importlib.metadata.PackageNotFoundError:
        return False

def install_packages(packages: list[tuple[str, str]], no_deps: bool = True) -> bool:
    import subprocess
    import sys
    
    for package, version in packages:
        my_cprint(f"Installing {package}=={version}...", "yellow")
        try:
            command = [sys.executable, "-m", "pip", "install", f"{package}=={version}"]
            if no_deps:
                command.append("--no-deps")
                
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            my_cprint(f"Successfully installed {package}=={version}", "green")
        except subprocess.CalledProcessError as e:
            my_cprint(f"Failed to install {package}: {e.stderr}", "red")
            return False

    return True

def check_and_install_dependencies(required_packages: dict[str, str], 
                                 backend_name: str = "backend",
                                 interactive: bool = True) -> bool:
    import sys

    missing_packages = []

    for package, version in required_packages.items():
        available, current_version = is_package_available(package)
        if not available:
            missing_packages.append((package, version))
        elif current_version != version:
            my_cprint(f"Warning: {package} version {current_version} found, expected {version}", "yellow")

    if not missing_packages:
        return True

    if not interactive or not sys.stdin.isatty():

        return False

    return False

def get_platform_info():
    import platform
    
    return {
        "system": platform.system(),
        "platform": platform.platform(),
        "architecture": platform.machine()
    }


def get_python_version():
    import sys
    
    major = sys.version_info.major
    minor = sys.version_info.minor
    
    return {
        'major': major,
        'minor': minor,
        'version_string': f'{major}.{minor}'
    }


def has_nvidia_gpu():
    import subprocess
    
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False



def gpu_summary():
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetMemoryInfo,
    )
    from numba import cuda

    cc_cores_per_SM = {
        (2, 0): 32,  (2, 1): 48,
        (3, 0): 192, (3, 5): 192, (3, 7): 192,
        (5, 0): 128, (5, 2): 128,
        (6, 0): 64,  (6, 1): 128,
        (7, 0): 64,  (7, 5): 64,
        (8, 0): 64,  (8, 6): 128, (8, 9): 128,
        (9, 0): 128,
        (10, 0): 128,
        (12, 0): 128,
    }

    nvmlInit()
    try:
        gpu_count = nvmlDeviceGetCount()
        summaries = []

        for idx in range(gpu_count):
            handle = nvmlDeviceGetHandleByIndex(idx)

            name = nvmlDeviceGetName(handle)
            vram_gb = nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3)

            dev = cuda.select_device(idx)
            cc_major, cc_minor = dev.compute_capability
            sm_count = dev.MULTIPROCESSOR_COUNT

            cores_per_sm = cc_cores_per_SM.get((cc_major, cc_minor), 128)
            total_cores = cores_per_sm * sm_count

            summaries.append(
                {
                    "index": idx,
                    "name": name,
                    "cuda_compute": f"{cc_major}.{cc_minor}",
                    "vram": round(vram_gb, 2),
                    "cuda_cores": total_cores,
                }
            )

        return summaries
    finally:
        nvmlShutdown()


def _needs_ocr_worker(path: str) -> bool:
    import fitz, logging
    try:
        with fitz.open(path) as doc:
            for page in doc:
                if page.get_text().strip():
                    return False
        return True
    except Exception as e:
        logging.error(f"PDF check error {path}: {e}")
        return False


def clean_triton_cache():
    import shutil
    from pathlib import Path

    triton_cache_dir = Path.home() / '.triton'

    if triton_cache_dir.exists():
        try:
            print(f"\nRemoving Triton cache at {triton_cache_dir}...")
            shutil.rmtree(triton_cache_dir)
            print("\033[92mTriton cache successfully removed.\033[0m")
            return True
        except Exception as e:
            print(f"\033[91mWarning: Failed to remove Triton cache: {e}\033[0m")
            return False
    else:
        print("\nNo Triton cache found to clean.")
        return True


def check_pdfs_for_ocr(script_dir):
    import multiprocessing as mp
    from pathlib import Path
    import fitz, logging, tempfile, os, threading
    from PySide6.QtWidgets import QMessageBox

    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or mp.cpu_count()
    except ImportError:
        logical = mp.cpu_count()
        estimate = max(logical // 2, logical - 4)
        physical = max(1, estimate)

    n_procs = max(1, physical - 1)

    docs_dir = Path(script_dir) / "Docs_for_DB"
    pdf_paths = [p for p in docs_dir.iterdir() if p.suffix.lower() == ".pdf"]
    if not pdf_paths:
        return True, ""

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_procs) as pool:
        mask = pool.map(_needs_ocr_worker, map(str, pdf_paths), chunksize=16)

    non_ocr_pdfs = [p for p, flag in zip(pdf_paths, mask) if flag]
    if non_ocr_pdfs:
        message = "The following PDF files appear to have no text content and likely need OCR done on them:\n\n"
        for pdf_path in non_ocr_pdfs:
            message += f"  - {pdf_path}\n"
        message += "\nPlease perform OCR on these by going to the Tools Tab first or remove them from the files selected for processing."

        msg_box = QMessageBox()
        msg_box.setWindowTitle("PDFs Need OCR")
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.addButton(QMessageBox.StandardButton.Ok)
        view_report_button = msg_box.addButton("View Report", QMessageBox.ButtonRole.ActionRole)
        msg_box.exec()

        if msg_box.clickedButton() == view_report_button:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                tmp.write("PDFs that need OCR:\n\n")
                for pdf_path in non_ocr_pdfs:
                    tmp.write(f"{pdf_path}\n")
                temp_path = tmp.name
            os.startfile(temp_path)

            def cleanup():
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
            threading.Timer(1.0, cleanup).start()

        return False, "PDFs without text content detected."

    return True, ""


class DownloadSignals(QObject):
    finished = Signal(bool, str)
    progress = Signal(str)


class DownloadRunnable(QRunnable):
    def __init__(self, download_func, *args):
        super().__init__()
        self.download_func = download_func
        self.args = args
        self.signals = DownloadSignals()

    def run(self):
        try:
            result = self.download_func(*self.args)
            self.signals.finished.emit(result, "Download completed successfully")
        except Exception as e:
            self.signals.finished.emit(False, str(e))


def download_with_threadpool(download_func, *args, callback=None):
    runnable = DownloadRunnable(download_func, *args)
    if callback:
        runnable.signals.finished.connect(callback)
    QThreadPool.globalInstance().start(runnable)


def download_kokoro_tts():
    from pathlib import Path
    from huggingface_hub import snapshot_download
    import shutil

    repo_id = "ctranslate2-4you/Kokoro-82M-light"
    tts_path = PROJECT_ROOT / "Models" / "tts" / "ctranslate2-4you--Kokoro-82M-light"

    try:
        tts_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading Kokoro TTS model from {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(tts_path),
            max_workers=4,
            token=False
        )
        print("Kokoro TTS model downloaded successfully")
        return True

    except Exception as e:
        print(f"Failed to download Kokoro TTS model: {e}")
        if tts_path.exists():
            shutil.rmtree(tts_path)
        return False


def normalize_chat_text(text):
    def split_num(num):
        num = num.group()
        if '.' in num:
            return num
        elif ':' in num:
            h, m = [int(n) for n in num.split(':')]
            if m == 0:
                return f"{h} o'clock"
            elif m < 10:
                return f'{h} oh {m}'
            return f'{h} {m}'
        year = int(num[:4])
        if year < 1100 or year % 1000 < 10:
            return num
        left, right = num[:2], int(num[2:4])
        s = 's' if num.endswith('s') else ''
        if 100 <= year % 1000 <= 999:
            if right == 0:
                return f'{left} hundred{s}'
            elif right < 10:
                return f'{left} oh {right}{s}'
        return f'{left} {right}{s}'

    def flip_money(m):
        m = m.group()
        bill = 'dollar' if m[0] == '$' else 'pound'
        if m[-1].isalpha():
            return f'{m[1:]} {bill}s'
        elif '.' not in m:
            s = '' if m[1:] == '1' else 's'
            return f'{m[1:]} {bill}{s}'
        b, c = m[1:].split('.')
        s = '' if b == '1' else 's'
        c = int(c.ljust(2, '0'))
        coins = f"cent{'' if c == 1 else 's'}" if m[0] == '$' else ('penny' if c == 1 else 'pence')
        return f'{b} {bill}{s} and {c} {coins}'

    def point_num(num):
        a, b = num.group().split('.')
        return ' point '.join([a, ' '.join(b)])

    text = text.replace('§', ' section ')
    
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)
    
    text = re.sub(r'\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)', split_num, text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b', flip_money, text)
    text = re.sub(r'\d*\.\d+', point_num, text)
    
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)

    text = text.strip()
    text = re.sub(r'^[^a-zA-Z]*', '', text)

    return text.strip()


def supports_flash_attention():
    logging.debug("Checking flash attention support")
    
    if not torch.cuda.is_available():
        logging.debug("CUDA not available, flash attention not supported")
        return False
        
    major, minor = torch.cuda.get_device_capability()
    logging.debug(f"CUDA compute capability: {major}.{minor}")
    
    supports = major >= 8
    logging.debug(f"Flash attention {'supported' if supports else 'not supported'}")
    return supports


def check_cuda_re_triton():
    logging.debug("Starting CUDA files check for Triton")
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_runtime = nvidia_base_path / 'cuda_runtime'
    
    logging.debug(f"Virtual environment base path: {venv_base}")
    logging.debug(f"NVIDIA base path: {nvidia_base_path}")
    logging.debug(f"CUDA runtime path: {cuda_runtime}")
    
    files_to_check = [
        cuda_runtime / "bin" / "cudart64_12.dll",
        cuda_runtime / "bin" / "ptxas.exe",
        cuda_runtime / "include" / "cuda.h",
        cuda_runtime / "lib" / "x64" / "cuda.lib"
    ]
    
    logging.debug("Beginning file existence checks")
    print("Checking CUDA files:")
    for file_path in files_to_check:
        exists = file_path.exists()
        status = "✓ Found" if exists else "✗ Missing"
        logging.debug(f"Checking {file_path}: {'exists' if exists else 'missing'}")
        print(f"{status}: {file_path}")
    print()
    logging.debug("CUDA file check completed")


def get_model_native_precision(embedding_model_name, vector_models):
    logging.debug(f"Looking for precision for model: {embedding_model_name}")
    model_name = os.path.basename(embedding_model_name)
    repo_style_name = model_name.replace('--', '/')
    
    for group_name, group_models in vector_models.items():
        logging.debug(f"Checking group: {group_name}")
        for model in group_models:
            logging.debug(f"Checking model: {model['repo_id']} / {model['name']}")
            if model['repo_id'] == repo_style_name or model['name'] in model_name:
                logging.debug(f"Found match! Using precision: {model['precision']}")
                return model['precision']
    logging.debug("No match found, defaulting to float32")
    return 'float32'


def get_appropriate_dtype(compute_device, use_half, model_native_precision):
    logging.debug(f"compute_device: {compute_device}")
    logging.debug(f"use_half: {use_half}")
    logging.debug(f"model_native_precision: {model_native_precision}")

    compute_device = compute_device.lower()
    model_native_precision = model_native_precision.lower()

    if compute_device == 'cpu':
        logging.debug("Using CPU, returning float32")
        return torch.float32

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_capability = torch.cuda.get_device_capability()
        logging.debug(f"CUDA is available. Capability: {cuda_capability}")
    else:
        cuda_capability = (0, 0)
        logging.debug("CUDA is not available.")

    if model_native_precision == 'bfloat16':
        if use_half:
            if cuda_available:
                if cuda_capability[0] >= 8:
                    logging.debug("Model native precision is bfloat16, GPU supports it, returning bfloat16")
                    return torch.bfloat16
                else:
                    logging.debug("GPU doesn't support bfloat16, falling back to float16")
                    return torch.float16
            else:
                logging.debug("No CUDA available for bfloat16, falling back to float32")
                return torch.float32
        else:
            logging.debug("Half checkbox not checked for bfloat16 model, returning float32")
            return torch.float32

    elif model_native_precision == 'float16':
        if use_half:
            if cuda_available:
                logging.debug("Model native precision is float16 and CUDA is available, returning float16")
                return torch.float16
            else:
                logging.debug("Model native precision is float16 but CUDA is not available, returning float32")
                return torch.float32
        else:
            logging.debug("Half checkbox not checked for float16 model, returning float32")
            return torch.float32

    elif model_native_precision == 'float32':
        if not use_half:
            logging.debug("Model is float32 and use_half is False, returning float32")
            return torch.float32
        else:
            if cuda_available:
                if cuda_capability[0] >= 8:
                    logging.debug("Using bfloat16 due to Ampere+ GPU")
                    return torch.bfloat16
                else:
                    logging.debug("Using float16 due to pre-Ampere GPU")
                    return torch.float16
            else:
                logging.debug("No CUDA available, returning float32")
                return torch.float32

    else:
        logging.debug(f"Unrecognized precision '{model_native_precision}', returning float32")
        return torch.float32

def format_citations(metadata_list):
    def group_metadata(metadata_list):
        grouped = {}
        for metadata in metadata_list:
            file_path = metadata['file_path']
            grouped.setdefault(file_path, {
                'name': Path(file_path).name,
                'scores': [],
                'pages': set(),
                'file_type': metadata.get('file_type', '')
            })
            grouped[file_path]['scores'].append(metadata['similarity_score'])
            if grouped[file_path]['file_type'] == '.pdf':
                page_number = metadata.get('page_number')
                if page_number is not None:
                    grouped[file_path]['pages'].add(page_number)
        return grouped

    def format_pages(pages):
        if not pages:
            return ''
        sorted_pages = sorted(pages)
        ranges = []
        start = prev = sorted_pages[0]
        for page in sorted_pages[1:]:
            if page == prev + 1:
                prev = page
            else:
                ranges.append((start, prev))
                start = prev = page
        ranges.append((start, prev))
        page_str = ', '.join(f"{s}-{e}" if s != e else f"{s}" for s, e in ranges)
        return f'<span style="color:#666;"> p.{page_str}</span>'

    def create_citation(data, file_path):
        min_score = min(data['scores'])
        max_score = max(data['scores'])
        score_range = f"{min_score:.4f}" if min_score == max_score else f"{min_score:.4f}-{max_score:.4f}"
        pages_html = format_pages(data['pages']) if data['file_type'] == '.pdf' else ''
        citation = (
            f'<a href="file:{file_path}" style="color:#DAA520;text-decoration:none;">{data["name"]}</a>'
            f'<span style="color:#808080;font-size:0.9em;"> ['
            f'<span style="color:#4CAF50;">{score_range}</span>]'
            f'{pages_html}'
            f'</span>'
        )
        return min_score, citation

    grouped_citations = group_metadata(metadata_list)
    citations_with_scores = [create_citation(data, file_path) for file_path, data in grouped_citations.items()]
    sorted_citations = [citation for _, citation in sorted(citations_with_scores)]
    list_items = "".join(f"<li>{citation}</li>" for citation in sorted_citations)

    return f"<ol>{list_items}</ol>"

def list_theme_files():
    script_dir = PROJECT_ROOT
    theme_dir = script_dir / 'CSS'
    return [f.name for f in theme_dir.iterdir() if f.suffix == '.css']

def load_stylesheet(filename):
    script_dir = PROJECT_ROOT
    stylesheet_path = script_dir / 'CSS' / filename
    with stylesheet_path.open('r') as file:
        stylesheet = file.read()
    return stylesheet

def ensure_theme_config():
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        if 'appearance' not in config:
            config['appearance'] = {}

        if 'theme' not in config['appearance'] or not config['appearance']['theme']:
            config['appearance']['theme'] = 'custom_stylesheet_default.css'

        with open('config.yaml', 'w') as f:
            yaml.safe_dump(config, f)

        return config['appearance']['theme']
    except Exception:
        return 'custom_stylesheet_default.css'

def update_theme_in_config(new_theme):
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        if 'appearance' not in config:
            config['appearance'] = {}

        config['appearance']['theme'] = new_theme

        with open('config.yaml', 'w') as f:
            yaml.safe_dump(config, f)
    except Exception:
        pass

def make_theme_changer(theme_name):
    def change_theme():
        QApplication.instance().setStyleSheet(load_stylesheet(theme_name))
        update_theme_in_config(theme_name)
    return change_theme

def backup_database(database_name=None):
   source_directory = Path('Vector_DB')
   backup_directory = Path('Vector_DB_Backup')

   if database_name:
       logging.debug("Starting incremental database backup")
       backup_directory.mkdir(parents=True, exist_ok=True)
       source_db_path = source_directory / database_name
       backup_db_path = backup_directory / database_name
       if backup_db_path.exists():
           try:
               shutil.rmtree(backup_db_path)
           except Exception as e:
               logging.debug(f"Failed to remove existing backup: {e}")
               print(f"Warning: Could not remove existing backup of {database_name}: {e}")
       try:
           shutil.copytree(source_db_path, backup_db_path)
           logging.debug(f"Successfully created backup of {database_name}")
       except Exception as e:
           logging.debug(f"Backup failed: {e}")
           print(f"Error backing up {database_name}: {e}")
   else:
       logging.debug("Starting full database backup")
       if backup_directory.exists():
           for item in backup_directory.iterdir():
               if item.is_dir():
                   shutil.rmtree(item)
               else:
                   item.unlink()
       else:
           backup_directory.mkdir(parents=True, exist_ok=True)
       shutil.copytree(source_directory, backup_directory, dirs_exist_ok=True)
       logging.debug("Database backup completed successfully")


def open_file(file_path):
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", file_path])
        else:
            subprocess.Popen(["xdg-open", file_path])
    except OSError:
        QMessageBox.warning(None, "Error", "No default viewer detected.")

def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError:
        QMessageBox.warning(None, "Unable to delete file(s), please delete manually.")

def check_preconditions_for_db_creation(script_dir, database_name, skip_ocr=False):
    if not database_name or len(database_name) < 3 or database_name.lower() in ["null", "none"]:
        return False, "Name must be at least 3 characters long and not be 'null' or 'none.'"

    vector_db_path = script_dir / "Vector_DB" / database_name
    if vector_db_path.exists():
        return False, (
            f"A vector database called '{database_name}' already exists—"
            "choose a different name or delete the old one first."
        )

    config_path = script_dir / 'config.yaml'
    if not config_path.exists():
        return False, "The configuration file (config.yaml) is missing."

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']
    documents_dir = script_dir / "Docs_for_DB"
    if platform.system() == "Darwin" and any(file.suffix in image_extensions for file in documents_dir.iterdir() if file.is_file()):
        return False, "Image processing has been disabled for MacOS until a fix can be implemented. Please remove all image files and try again."

    embedding_model_name = config.get('EMBEDDING_MODEL_NAME')
    if not embedding_model_name:
        return False, "You must first download an embedding model, select it, and choose documents before proceeding."

    if not any(file.is_file() for file in documents_dir.iterdir()):
        return False, "No documents are yet added to be processed."

    compute_device = config.get('Compute_Device', {}).get('available', [])
    database_creation = config.get('Compute_Device', {}).get('database_creation')
    if ("cuda" in compute_device or "mps" in compute_device) and database_creation == "cpu":
        return False, ("GPU-acceleration is available and strongly recommended. "
                       "Please switch the database creation device to 'cuda' or 'mps', "
                       "or confirm your choice in the GUI.")

    if not torch.cuda.is_available():
        if config.get('database', {}).get('half', False):
            message = ("CUDA is not available on your system, but half-precision (FP16) "
                       "is selected for database creation. Half-precision requires CUDA. "
                       "Please disable half-precision in the configuration or use a CUDA-enabled GPU.")
            return False, message

    if not skip_ocr:
        ocr_check, ocr_message = check_pdfs_for_ocr(script_dir)
        if not ocr_check:
            return False, ocr_message

    return True, ""


def check_preconditions_for_submit_question(script_dir):
    config_path = script_dir / 'config.yaml'

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    database_to_search = config.get('database', {}).get('database_to_search')

    vector_db_subdir = script_dir / "Vector_DB" / str(database_to_search) if database_to_search else None

    return True, ""

def my_cprint(*args, **kwargs):
    filename = os.path.basename(sys._getframe(1).f_code.co_filename)
    modified_message = f"{args[0]}"
    kwargs['flush'] = True
    cprint(modified_message, *args[1:], **kwargs)

def has_bfloat16_support():
   logging.debug("Checking bfloat16 support")

   if not torch.cuda.is_available():
       logging.debug("CUDA not available, bfloat16 not supported")
       return False

   capability = torch.cuda.get_device_capability()
   logging.debug(f"CUDA compute capability: {capability}")

   has_support = capability >= (8, 0)
   logging.debug(f"bfloat16 {'supported' if has_support else 'not supported'}")
   return has_support

def set_logging_level():
    library_levels = {
        "accelerate": logging.WARNING,
        "bitsandbytes": logging.WARNING,
        "ctranslate2": logging.WARNING,
        "datasets": logging.WARNING,
        "einops": logging.WARNING,
        "einx": logging.WARNING,
        "flash_attn": logging.WARNING,
        "huggingface-hub": logging.WARNING,
        "langchain": logging.WARNING,
        "langchain-community": logging.WARNING,
        "langchain-core": logging.WARNING,
        "langchain-huggingface": logging.WARNING,
        "langchain-text-splitters": logging.WARNING,
        "numpy": logging.WARNING,
        "openai": logging.WARNING,
        "openai-whisper": logging.WARNING,
        "optimum": logging.WARNING,
        "pillow": logging.WARNING,
        "requests": logging.WARNING,
        "sentence-transformers": logging.WARNING,
        "sounddevice": logging.WARNING,
        "speechbrain": logging.WARNING,
        "sympy": logging.WARNING,
        "tiledb": logging.WARNING,
        "tiledb-cloud": logging.WARNING,
        "tiledb-vector-search": logging.WARNING,
        "timm": logging.WARNING,
        "tokenizers": logging.WARNING,
        "torch": logging.WARNING,
        "torchaudio": logging.WARNING,
        "torchvision": logging.WARNING,
        "transformers": logging.WARNING,
        "unstructured": logging.WARNING,
        "unstructured-client": logging.WARNING,
        "vector-quantize-pytorch": logging.WARNING,
        "vocos": logging.WARNING,
        "xformers": logging.WARNING
    }

    for lib, level in library_levels.items():
        logging.getLogger(lib).setLevel(level)

def prepare_long_path(base_path: str, filename: str) -> str:
    base_path = os.path.normpath(base_path)
    full_path = os.path.join(base_path, filename)
    
    if os.name == 'nt' and len(full_path) > 255:
        full_path = "\\\\?\\" + os.path.abspath(full_path)
    
    return full_path
