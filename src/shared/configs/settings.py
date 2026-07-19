from pathlib import Path

from pydantic import Field, SecretStr
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


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHAT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    provider: str = Field(default="ollama")
    model: str = Field(default="qwen2.5:1.5b")
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=120.0, ge=0)
    top_k: int = Field(default=8, ge=1, le=100)


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGESTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    processor: str = Field(default="unstructured")
    extraction_timeout_seconds: float = Field(default=120.0, gt=0)
    pdf_strategy: str = Field(default="fast")
    max_file_size_bytes: int = Field(default=10_485_760, ge=1)
    chunk_max_characters: int = Field(default=1500, ge=128)
    chunk_combine_text_under_n_chars: int = Field(default=256, ge=0)
    chunk_new_after_n_chars: int = Field(default=1000, ge=128)


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_k: int = Field(default=50, ge=1, le=500)
    num_candidates: int = Field(default=100, ge=1, le=2000)
    rank_constant: int = Field(default=60, ge=1)
    min_fused_score: float | None = Field(default=None, ge=0.0)


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    url: SecretStr = Field(
        default=SecretStr(
            "postgresql+asyncpg://postgres:postgres@localhost:5432/rag-sys"
        )
    )
    echo: bool = Field(default=False)
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)


class ObjectStorageSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OBJECT_STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    endpoint: str = Field(default="localhost:9000")
    access_key: SecretStr = Field(default=SecretStr("minioadmin"))
    secret_key: SecretStr = Field(default=SecretStr("minioadmin"))
    secure: bool = False
    region: str = Field(default="us-east-1")


class KafkaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KAFKA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    bootstrap_servers: str = Field(default="localhost:9092")
    client_id: str = Field(default="rag-sys")
    session_timeout_ms: int = Field(default=45_000, ge=1_000)
    max_poll_interval_ms: int = Field(default=1_800_000, ge=1_000)
    consumer_poll_timeout_ms: int = Field(default=1_000, ge=100)


class SnowflakeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SNOWFLAKE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    instance_id: int = Field(default=0, ge=0, le=1023)


class DocumentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCUMENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    max_upload_size_bytes: int = Field(
        default=10_485_760,
        gt=0,
    )
    upload_url_ttl_seconds: int = Field(
        default=900,
        ge=60,
        le=3600,
    )
    allowed_mime_types: frozenset[str] = Field(
        default=frozenset(
            {
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "image/jpeg",
                "image/png",
            },
        ),
    )


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
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    object_storage: ObjectStorageSettings = Field(default_factory=ObjectStorageSettings)
    snowflake: SnowflakeSettings = Field(default_factory=SnowflakeSettings)
    document: DocumentSettings = Field(default_factory=DocumentSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
