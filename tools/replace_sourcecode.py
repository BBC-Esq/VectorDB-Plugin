import hashlib
from pathlib import Path
import shutil
import sys
import zipfile

class DependencyUpdater:
    def __init__(self):
        self.site_packages_path = self.get_site_packages_path()

    def get_site_packages_path(self):
        paths = sys.path
        site_packages_paths = [Path(path) for path in paths if 'site-packages' in path.lower()]
        return site_packages_paths[0] if site_packages_paths else None

    def find_dependency_path(self, dependency_path_segments):
        current_path = self.site_packages_path
        if current_path and current_path.exists():
            for segment in dependency_path_segments:
                next_path = next((current_path / child for child in current_path.iterdir() if child.name.lower() == segment.lower()), None)
                if next_path is None:
                    return None
                current_path = next_path
            return current_path
        return None

    @staticmethod
    def hash_file(filepath):
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        return hasher.hexdigest()

    @staticmethod
    def copy_and_overwrite_if_necessary(source_path, target_path):
        if not target_path.exists() or DependencyUpdater.hash_file(source_path) != DependencyUpdater.hash_file(target_path):
            shutil.copy(source_path, target_path)
            DependencyUpdater.print_status("SUCCESS", f"{source_path} has been successfully copied to {target_path}.")
        else:
            DependencyUpdater.print_status("SKIP", f"{target_path} is already up to date.")

    def update_file_in_dependency(self, source_folder, file_name, dependency_path_segments):
        target_path = self.find_dependency_path(dependency_path_segments)
        if target_path is None:
            self.print_status("ERROR", "Target dependency path not found.")
            return

        source_path = Path(__file__).parent / source_folder / file_name
        if not source_path.exists():
            self.print_status("ERROR", f"{file_name} not found in {source_folder}.")
            return

        target_file = None
        for child in target_path.iterdir():
            if child.is_file() and child.name.lower() == file_name.lower():
                target_file = child
                break

        if target_file:
            target_file_path = target_file
        else:
            target_file_path = target_path / file_name
        self.copy_and_overwrite_if_necessary(source_path, target_file_path)

    @staticmethod
    def print_status(status, message):
        colors = {
            "SUCCESS": "\033[92m",  # Green
            "SKIP": "\033[93m",     # Yellow
            "ERROR": "\033[91m",    # Red
            "INFO": "\033[94m"      # Blue
        }
        reset_color = "\033[0m"
        print(f"{colors.get(status, reset_color)}[{status}] {message}{reset_color}")

    @staticmethod
    def print_ascii_table(title, rows):
        table_width = max(len(title), max(len(row) for row in rows)) + 4
        border = f"+{'-' * (table_width - 2)}+"
        print(border)
        print(f"| {title.center(table_width - 4)} |")
        print(border)
        for row in rows:
            print(f"| {row.ljust(table_width - 4)} |")
        print(border)

def replace_sentence_transformer_file():
    updater = DependencyUpdater()
    updater.update_file_in_dependency("Assets", "SentenceTransformer.py", ["sentence_transformers"])

def replace_chattts_file():
    updater = DependencyUpdater()
    updater.update_file_in_dependency("Assets", "core.py", ["ChatTTS"])

def add_cuda_files():
    updater = DependencyUpdater()

    updater.print_ascii_table("CUDA FILES UPDATE", ["Copying ptxas.exe", "Extracting cudart_lib.zip"])

    source_path = updater.find_dependency_path(["nvidia", "cuda_nvcc", "bin"])
    if source_path is None:
        updater.print_status("ERROR", "Source path for ptxas.exe not found.")
        return

    source_file = source_path / "ptxas.exe"
    if not source_file.exists():
        updater.print_status("ERROR", "ptxas.exe not found in the source directory.")
        return

    target_path = updater.find_dependency_path(["nvidia", "cuda_runtime", "bin"])
    if target_path is None:
        updater.print_status("ERROR", "Target path (cuda_runtime) not found.")
        return

    target_file = target_path / "ptxas.exe"
    updater.copy_and_overwrite_if_necessary(source_file, target_file)

    zip_path = Path(__file__).parent / "Assets" / "cudart_lib.zip"
    if not zip_path.exists():
        updater.print_status("ERROR", "cudart_lib.zip not found.")
        return

    cuda_lib_runtime_path = target_path.parent
    if target_path is None or not target_path.exists():
        updater.print_status("ERROR", "Parent directory of cuda_runtime/bin not found.")
        return

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cuda_lib_runtime_path)
            updater.print_status("SUCCESS", f"Extracted cudart_lib.zip to {cuda_lib_runtime_path}")
    except zipfile.BadZipFile:
        updater.print_status("ERROR", "cudart_lib.zip is corrupted or not a zip file.")
    except PermissionError:
        updater.print_status("ERROR", "Permission denied when extracting cudart_lib.zip.")
    except Exception as e:
        updater.print_status("ERROR", f"Unexpected error during extraction: {str(e)}")

def setup_vector_db():
    updater = DependencyUpdater()

    zip_path = Path(__file__).parent / "Assets" / "user_manual_db.zip"
    if not zip_path.exists():
        updater.print_status("ERROR", "user_manual_db.zip not found in Assets folder.")
        return

    vector_db_path = Path(__file__).parent / "Vector_DB"
    vector_db_backup_path = Path(__file__).parent / "Vector_DB_Backup"

    try:
        vector_db_path.mkdir(exist_ok=True)
        vector_db_backup_path.mkdir(exist_ok=True)
    except PermissionError:
        updater.print_status("ERROR", "Insufficient permissions to create directories.")
        return
    except Exception as e:
        updater.print_status("ERROR", f"Error creating directories: {str(e)}")
        return

    user_manual_paths = [
        vector_db_path / "user_manual",
        vector_db_backup_path / "user_manual"
    ]

    for path in user_manual_paths:
        if path.exists():
            try:
                shutil.rmtree(path, ignore_errors=False)
                updater.print_status("INFO", f"Removed existing user_manual folder from {path.parent}")
            except PermissionError:
                updater.print_status("ERROR", f"Permission denied when trying to remove {path}")
                return
            except Exception as e:
                updater.print_status("ERROR", f"Error removing {path}: {str(e)}")
                return

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if zip_ref.testzip() is not None:
                updater.print_status("ERROR", "Zip file is corrupted.")
                return
            zip_ref.extractall(vector_db_path)
            zip_ref.extractall(vector_db_backup_path)
        updater.print_status("SUCCESS", f"Successfully extracted user_manual_db.zip to {vector_db_path} and {vector_db_backup_path}")
    except PermissionError:
        updater.print_status("ERROR", "Permission denied when extracting zip file.")
    except Exception as e:
        updater.print_status("ERROR", f"Error extracting zip file: {str(e)}")

def check_embedding_model_dimensions():
    import yaml
    updater = DependencyUpdater()
    config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        updater.print_status("ERROR", "config.yaml not found in current directory.")
        return

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        if config is None:
            config = {}

        if 'EMBEDDING_MODEL_DIMENSIONS' not in config:
            config['EMBEDDING_MODEL_DIMENSIONS'] = None
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            updater.print_status("SUCCESS", "Added EMBEDDING_MODEL_DIMENSIONS: null to config.yaml")
        else:
            updater.print_status("SKIP", "EMBEDDING_MODEL_DIMENSIONS already exists in config.yaml")

    except yaml.YAMLError as e:
        updater.print_status("ERROR", f"Error parsing config.yaml: {str(e)}")
    except Exception as e:
        updater.print_status("ERROR", f"Unexpected error while processing config.yaml: {str(e)}")

if __name__ == "__main__":
    DependencyUpdater.print_ascii_table("DEPENDENCY UPDATER", [
        "Replace Sentence Transformer File",
        "Replace ChatTTS File",
        "Add CUDA Files",
        "Setup Vector DB",
        "Check Config EMBEDDING_MODEL_DIMENSIONS"
    ])

    replace_sentence_transformer_file()
    replace_chattts_file()
    add_cuda_files()
    setup_vector_db()
    check_embedding_model_dimensions()