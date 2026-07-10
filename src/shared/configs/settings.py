from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOGGING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    level: str | None = Field(default="INFO")
    format: str | None = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    datefmt: str | None = Field(default="%Y-%m-%d %H:%M:%S")
    handlers: list[str] | None = Field(default=["console", "file"])
    root_logger: bool | None = Field(default=True)
    root_level: str | None = Field(default="INFO")
    file_path: Path = Field(default=Path("logs/app.log"))
    file_max_bytes: int = Field(default=1_048_576, ge=1)
    file_backup_count: int = Field(default=5, ge=0)


class ElasticsearchSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ELASTICSEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    urls: list[str] = Field(default=["http://localhost:9200"])
    index: str = Field(default="rag-documents")
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    number_of_shards: int = Field(default=1, ge=1)
    number_of_replicas: int = Field(default=0, ge=0)

class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    url: str = Field(default="http://localhost:11434")

class VLLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    embedding_url: str = Field(default="http://localhost:8001")
    rerank_url: str = Field(default="http://localhost:8002")
    chat_url: str = Field(default="http://localhost:8003")

class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    provider: str = Field(default="ollama")
    dimensions: int = Field(default=1024, ge=32, le=1024)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    timeout_seconds: float = Field(default=120.0, ge=0)
    model: str = Field(default="batiai/qwen3-embedding:0.6b")

class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGESTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    max_file_size_bytes: int = Field(default=10_485_760, ge=1)
    chunk_max_characters: int = Field(default=1500, ge=128)
    chunk_combine_text_under_n_chars: int = Field(default=256, ge=0)
    chunk_new_after_n_chars: int = Field(default=1000, ge=128)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)