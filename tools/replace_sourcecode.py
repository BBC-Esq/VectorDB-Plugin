import hashlib
from pathlib import Path
import shutil
import sys
import zipfile

from core.constants import PROJECT_ROOT

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
            target_path.unlink(missing_ok=True)
            shutil.copy(source_path, target_path)
            DependencyUpdater.print_status("SUCCESS", f"{source_path} has been successfully copied to {target_path}.")
        else:
            DependencyUpdater.print_status("SKIP", f"{target_path} is already up to date.")

    def update_file_in_dependency(self, source_folder, file_name, dependency_path_segments):
        target_path = self.find_dependency_path(dependency_path_segments)
        if target_path is None:
            self.print_status("ERROR", "Target dependency path not found.")
            return

        source_path = PROJECT_ROOT / source_folder / file_name
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
            "SUCCESS": "\033[92m",
            "SKIP": "\033[93m",
            "ERROR": "\033[91m",
            "INFO": "\033[94m"
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

def replace_chattts_file():
    updater = DependencyUpdater()
    updater.update_file_in_dependency("Assets", "core.py", ["ChatTTS"])

def setup_vector_db():
    updater = DependencyUpdater()

    zip_path = PROJECT_ROOT / "Assets" / "user_manual_db.zip"
    if not zip_path.exists():
        updater.print_status("ERROR", "user_manual_db.zip not found in Assets folder.")
        return

    vector_db_path = PROJECT_ROOT / "Vector_DB"
    vector_db_backup_path = PROJECT_ROOT / "Vector_DB_Backup"

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
    config_path = PROJECT_ROOT / "config.yaml"

    if not config_path.exists():
        updater.print_status("ERROR", "config.yaml not found in current directory.")
        return

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        if config is None:
            config = {}

        if 'EMBEDDING_MODEL_DIMENSIONS' not in config:
            config['EMBEDDING_MODEL_DIMENSIONS'] = None
            with open(config_path, 'w', encoding='utf-8') as file:
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
        "Replace ChatTTS File",
        "Setup Vector DB",
        "Check Config EMBEDDING_MODEL_DIMENSIONS"
    ])

    replace_chattts_file()
    setup_vector_db()
    check_embedding_model_dimensions()
