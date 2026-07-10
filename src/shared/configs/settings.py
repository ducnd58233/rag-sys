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
    file_max_bytes: int = Field(default=1024, ge=1)
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    elasticsearch: ElasticsearchSettings = Field(default_factory=ElasticsearchSettings)
