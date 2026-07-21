from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOGGING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    level: str | None = Field(default="INFO")
    formatter: str = Field(default="json")
    text_format: str | None = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    datefmt: str | None = Field(default="%Y-%m-%d %H:%M:%S")
    handlers: list[str] | None = Field(default=["console", "file"])
    root_logger: bool | None = Field(default=True)
    root_level: str | None = Field(default="INFO")
    file_path: Path = Field(default=Path("logs/app.log"))
    file_max_bytes: int = Field(default=1_048_576, ge=1)
    file_backup_count: int = Field(default=5, ge=0)


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    sdk_disabled: bool = Field(default=False)
    exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    service_version: str = Field(default="0.1.0")
    environment: str = Field(default="local")
    metric_export_interval_ms: int = Field(default=15_000, ge=1_000)

    @property
    def enabled(self) -> bool:
        return not self.sdk_disabled


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
    max_retries: int = Field(default=3, ge=0)
    retry_on_timeout: bool = Field(default=True)
    retry_backoff_base: float = Field(default=2.0, gt=0)
    retry_backoff_cap: float = Field(default=30.0, gt=0)
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
    timeout_seconds: float = Field(default=120.0, ge=0)
    top_k: int = Field(default=8, ge=1, le=100)
    complex_rag_enabled: bool = Field(default=True)
    complex_rag_max_retrieval_queries: int = Field(default=20, ge=1, le=50)
    complex_rag_per_query_top_k: int = Field(default=6, ge=1, le=50)
    complex_rag_final_top_k: int = Field(default=12, ge=1, le=100)
    complex_rag_max_iterations: int = Field(default=2, ge=1, le=3)


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
    outbox_relay_enabled: bool = Field(default=True)
    outbox_relay_interval_seconds: float = Field(default=5.0, gt=0)


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


class RoutingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROUTING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    llm_router_timeout_seconds: float = Field(default=30.0, gt=0)
    temporal_recency_half_life_days: float = Field(default=90.0, gt=0)
    graph_max_hops: int = Field(default=2, ge=1, le=4)


class GraphDbSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRAPHDB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: SecretStr = Field(default=SecretStr("rag-sys-dev"))
    database: str = Field(default="neo4j")
    extraction_max_characters: int = Field(default=12_000, ge=1)


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
    pool_pre_ping: bool = Field(default=True)
    pool_recycle_seconds: int = Field(default=1800, ge=-1)


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
    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_base_delay_seconds: float = Field(default=2.0, gt=0)
    retry_max_delay_seconds: float = Field(default=30.0, gt=0)


class SnowflakeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SNOWFLAKE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    instance_id: int = Field(default=0, ge=0, le=1023)


class ResilienceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESILIENCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    retry_max_attempts: int = Field(default=3, ge=1, le=10)
    retry_base_delay_seconds: float = Field(default=0.5, gt=0)
    retry_max_delay_seconds: float = Field(default=8.0, gt=0)
    warmup_enabled: bool = Field(default=True)
    warmup_embedding: bool = Field(default=True)
    warmup_chat: bool = Field(default=True)
    warmup_embedding_text: str = Field(default="warmup")
    warmup_chat_system: str = Field(default="Reply with pong only.")
    warmup_chat_user: str = Field(default="ping")
    warmup_fail_fast: bool = Field(default=True)


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
                "text/markdown",
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
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    object_storage: ObjectStorageSettings = Field(default_factory=ObjectStorageSettings)
    snowflake: SnowflakeSettings = Field(default_factory=SnowflakeSettings)
    document: DocumentSettings = Field(default_factory=DocumentSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    graphdb: GraphDbSettings = Field(default_factory=GraphDbSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)

    @model_validator(mode="after")
    def _validate_kafka_max_poll_interval(self) -> "Settings":
        extraction_timeout_ms = self.ingestion.extraction_timeout_seconds * 1000
        if self.kafka.max_poll_interval_ms <= extraction_timeout_ms:
            raise ValueError(
                "KAFKA_MAX_POLL_INTERVAL_MS must exceed "
                "INGESTION_EXTRACTION_TIMEOUT_SECONDS * 1000 to avoid "
                "consumer group rebalance loops"
            )
        return self
