from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import disable_progress_bars, RepositoryNotFoundError, GatedRepoError
from huggingface_hub.hf_api import RepoFile
from PySide6.QtCore import QObject, Signal
import fnmatch
import humanfriendly
import atexit
import yaml
import functools

class ModelDownloadedSignal(QObject):
    downloaded = Signal(str, str)

model_downloaded_signal = ModelDownloadedSignal()

MODEL_DIRECTORIES = {
    "vector": "vector",
    "chat": "chat",
    "tts": "tts",
    "jeeves": "jeeves",
    "ocr": "ocr"
}

@functools.lru_cache(maxsize=1)
def get_hf_token():
    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            token = (data.get("hf_access_token") or "").strip()
            return token or None
        except Exception:
            return None
    return None

class ModelDownloader(QObject):
    def __init__(self, model_info, model_type):
        super().__init__()
        self.model_info = model_info
        self.model_type = model_type
        self._model_directory = None
        self.hf_token = get_hf_token()
        self.api = HfApi(token=False)
        self.api.timeout = 60
        disable_progress_bars()
        self.local_dir = self.get_model_directory()

    def cleanup_incomplete_download(self):
        try:
            if hasattr(self, "local_dir") and self.local_dir and self.local_dir.exists():
                if not any(self.local_dir.iterdir()):
                    import shutil
                    shutil.rmtree(self.local_dir)
        except Exception:
            pass

    def get_model_directory_name(self):
        repo_id = self.get_model_url()
        if isinstance(repo_id, str):
            return repo_id.replace("/", "--")
        return str(repo_id)

    def get_model_directory(self):
        base = Path("Models")
        sub = MODEL_DIRECTORIES.get(self.model_type, self.model_type)
        return base / sub / self.get_model_directory_name()

    def get_model_url(self):
        if isinstance(self.model_info, dict):
            return self.model_info.get("repo_id") or self.model_info.get("url") or self.model_info.get("name")
        return self.model_info

    def check_repo_type(self, repo_id):
        try:
            repo_info = self.api.repo_info(repo_id, timeout=60, token=False)
            if getattr(repo_info, "private", False):
                return "private"
            if getattr(repo_info, "gated", False):
                return "gated"
            return "public"
        except RepositoryNotFoundError:
            return "not_found"
        except GatedRepoError:
            return "gated"
        except Exception as e:
            msg = str(e).lower()
            if "401" in msg or "403" in msg or "gated" in msg:
                try:
                    api_with_token = HfApi(token=self.hf_token or False)
                    _ = api_with_token.repo_info(repo_id, timeout=60)
                    return "public"
                except GatedRepoError:
                    return "gated"
                except Exception:
                    return "gated" if not self.hf_token else "error"
            return "error"

    def _list_repo_files(self, repo_id, use_token):
        api = self.api if not use_token else HfApi(token=self.hf_token)
        return list(api.list_repo_tree(repo_id, recursive=True))

    def _select_patterns(self, repo_files, allow_patterns, ignore_patterns):
        final_ignore = [
            "*.ckpt",
            "*.onnx",
            "*.h5",
            "*.tflite",
            "*.pb",
            "*.msgpack",
            "*.safetensors.index.json",
            "*.bin.index.json",
            "*.flax",
            "*.npz",
            "*.tar",
            "*.tar.gz",
            "*.zip",
            "*.rar",
            "*.7z",
            "*.gz",
            "*.bz2",
            "*.xz",
            "*.md",
            "README*",
            "LICENSE*",
            ".*",
            ".gitattributes",
            ".git*",
        ]
        if ignore_patterns:
            final_ignore.extend(ignore_patterns)
        safetensors_files = [f.rfilename for f in repo_files if isinstance(f, RepoFile) and f.rfilename.endswith(".safetensors")]
        bin_files = [f.rfilename for f in repo_files if isinstance(f, RepoFile) and f.rfilename.endswith(".bin")]
        if safetensors_files and bin_files:
            final_ignore.append("*.bin")
        if safetensors_files or bin_files:
            final_ignore.append("*consolidated*")
        if allow_patterns is None:
            allow_patterns = ["*.json", "*.safetensors", "*.bin", "*.model", "tokenizer*", "vocab*", "merges.txt", "config.yaml", "modules.json", "1_Pooling/*", "sentencepiece.*", "spiece.*"]
        return allow_patterns, final_ignore

    def _filter_and_size(self, repo_files, allow_patterns, ignore_patterns):
        included_files = []
        ignored_files = []
        total_size = 0
        for file in repo_files:
            if not isinstance(file, RepoFile):
                continue
            path = file.rfilename
            if any(fnmatch.fnmatch(path, pat) for pat in ignore_patterns):
                ignored_files.append(path)
                continue
            if allow_patterns and not any(fnmatch.fnmatch(path, pat) for pat in allow_patterns):
                ignored_files.append(path)
                continue
            included_files.append(path)
            try:
                if file.size is not None:
                    total_size += int(file.size)
            except Exception:
                pass
        return included_files, ignored_files, total_size

    def download(self, allow_patterns=None, ignore_patterns=None):
        repo_id = self.get_model_url()
        repo_type = self.check_repo_type(repo_id)
        if repo_type not in ["public", "gated"]:
            if repo_type == "private":
                print(f"Repository {repo_id} is private and requires a token.")
                if not self.hf_token:
                    print("No Hugging Face token found. Add one in config.yaml.")
                return
            if repo_type == "not_found":
                print(f"Repository {repo_id} not found.")
                return
            print(f"Error checking repository {repo_id}.")
            return
        if repo_type == "gated" and not self.hf_token:
            print(f"Repository {repo_id} is gated and requires access and a token.")
            return
        local_dir = self.get_model_directory()
        local_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.cleanup_incomplete_download)
        try:
            repo_files = self._list_repo_files(repo_id, use_token=(repo_type == "gated"))
            allow_patterns, final_ignore_patterns = self._select_patterns(repo_files, allow_patterns, ignore_patterns)
            included_files, ignored_files, total_size = self._filter_and_size(repo_files, allow_patterns, final_ignore_patterns)
            readable_total_size = humanfriendly.format_size(total_size, binary=True)
            print(f"\nTotal size to be downloaded: {readable_total_size}")
            print("\nFiles to be downloaded:")
            for f in included_files:
                print(f"- {f}")
            print(f"\nDownloading to {local_dir}...")
            download_kwargs = {
                "repo_id": repo_id,
                "local_dir": str(local_dir),
                "max_workers": 8,
                "local_dir_use_symlinks": False,
                "ignore_patterns": final_ignore_patterns,
                "allow_patterns": allow_patterns,
                "etag_timeout": 60
            }
            if repo_type == "gated" and self.hf_token:
                download_kwargs["token"] = self.hf_token
            else:
                download_kwargs["token"] = False
            snapshot_download(**download_kwargs)
            print("\033[92mModel downloaded and ready to use.\033[0m")
            atexit.unregister(self.cleanup_incomplete_download)
            model_downloaded_signal.downloaded.emit(self.get_model_directory_name(), self.model_type)
        except Exception as e:
            print(f"An error occurred during download: {str(e)}")
            if local_dir.exists():
                import shutil
                shutil.rmtree(local_dir)

def download_embedding_model(repo_id, local_dir=None):
    info = {"repo_id": repo_id}
    downloader = ModelDownloader(info, "vector")
    if local_dir:
        downloader._model_directory = Path(local_dir)
        downloader.local_dir = downloader.get_model_directory()
    downloader.download()

def download_chat_model(repo_id, local_dir=None):
    info = {"repo_id": repo_id}
    downloader = ModelDownloader(info, "chat")
    if local_dir:
        downloader._model_directory = Path(local_dir)
        downloader.local_dir = downloader.get_model_directory()
    downloader.download()
