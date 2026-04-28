from pathlib import Path
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
import threading


class OpenAIConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    reasoning_effort: str = "medium"


class MiniMaxConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "MiniMax-M2.7"


class ServerConfig(BaseModel):
    api_key: str = ""
    connection_str: str = "http://127.0.0.1:1234/v1"
    show_thinking: bool = False

    @field_validator('connection_str')
    @classmethod
    def validate_connection_str(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Connection string must start with http:// or https://")
        return v


class DatabaseConfig(BaseModel):
    chunk_size: int = Field(default=700, gt=0, le=100000)
    chunk_overlap: int = Field(default=250, ge=0, le=100000)
    contexts: int = Field(default=5, gt=0, le=1000)
    similarity: float = Field(default=0.7, ge=0.0, le=1.0)
    half: bool = False
    database_to_search: str = ""
    document_types: str = ""
    search_term: str = ""
    pipeline_preset: str = "normal"

    @field_validator('contexts', mode='before')
    @classmethod
    def coerce_contexts(cls, v):
        if isinstance(v, str):
            return int(v)
        return v

    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator('pipeline_preset')
    @classmethod
    def validate_pipeline_preset(cls, v: str) -> str:
        valid = {"minimal", "low", "normal", "high", "maximum"}
        if v not in valid:
            raise ValueError(f"pipeline_preset must be one of {valid}")
        return v


class ComputeDeviceConfig(BaseModel):
    available: list = Field(default_factory=lambda: ["cpu"])
    database_creation: str = "cpu"
    database_query: str = "cpu"
    gpu_brand: Optional[str] = None

    @field_validator('database_creation', 'database_query')
    @classmethod
    def validate_device(cls, v: str, info) -> str:
        if 'available' in info.data and v not in info.data['available']:
            return "cpu"
        return v


class DatabaseInfo(BaseModel):
    model: str
    chunk_size: int
    chunk_overlap: int


class AppearanceConfig(BaseModel):
    theme: str = "default"


class PlatformInfo(BaseModel):
    os: str = ""


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        extra='allow',
        validate_assignment=True
    )

    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    minimax: MiniMaxConfig = Field(default_factory=MiniMaxConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    Compute_Device: ComputeDeviceConfig = Field(default_factory=ComputeDeviceConfig)
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    Platform_Info: PlatformInfo = Field(default_factory=PlatformInfo)

    EMBEDDING_MODEL_NAME: Optional[str] = None
    EMBEDDING_MODEL_DIMENSIONS: Optional[int] = None
    hf_access_token: Optional[str] = None
    created_databases: Dict[str, DatabaseInfo] = Field(default_factory=dict)

    _config_path: Path = PrivateAttr(default=Path("config.yaml"))
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    @property
    def root_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def docs_dir(self) -> Path:
        return self.root_dir / "Docs_for_DB"

    @property
    def vector_db_dir(self) -> Path:
        return self.root_dir / "Vector_DB"

    @property
    def vector_db_backup_dir(self) -> Path:
        return self.root_dir / "Vector_DB_Backup"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "Models"

    @property
    def vector_models_dir(self) -> Path:
        return self.models_dir / "vector"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        config_path = path or Path("config.yaml")
        if not config_path.exists():
            instance = cls()
            instance._config_path = config_path
            instance.save(config_path)
            return instance
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            if 'created_databases' in data:
                for db_name, db_data in data['created_databases'].items():
                    if isinstance(db_data, dict) and not isinstance(db_data, DatabaseInfo):
                        data['created_databases'][db_name] = DatabaseInfo(**db_data)
            instance = cls(**data)
            instance._config_path = config_path
            return instance
        except Exception as e:
            print(f"Error loading config: {e}")
            instance = cls()
            instance._config_path = config_path
            return instance

    def save(self, path: Optional[Path] = None) -> None:
        save_path = path or self._config_path
        with self._lock:
            data = self.model_dump()
            temp_path = save_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True)
            temp_path.replace(save_path)

    def update_field(self, field_path: str, value: Any) -> None:
        with self._lock:
            parts = field_path.split('.')
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
            self.save()

    def update_setting(self, field_path: str, value: Any) -> tuple[bool, str]:
        try:
            self.update_field(field_path, value)
            return True, "Setting updated successfully"
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Error updating setting: {str(e)}"

    def add_database(self, name: str, model_path: str, chunk_size: int, chunk_overlap: int) -> None:
        self.created_databases[name] = DatabaseInfo(
            model=model_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.save()

    def remove_database(self, name: str) -> None:
        if name in self.created_databases:
            del self.created_databases[name]
            self.save()

    def get_user_databases(self) -> list[str]:
        return [name for name in self.created_databases.keys() if name != "user_manual"]


_config_instance: Optional[AppConfig] = None
_config_lock = threading.Lock()


def get_config() -> AppConfig:
    global _config_instance
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = AppConfig.load()
    return _config_instance


def reload_config() -> AppConfig:
    global _config_instance
    with _config_lock:
        _config_instance = AppConfig.load()
    return _config_instance
