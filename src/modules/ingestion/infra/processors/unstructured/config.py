from dataclasses import dataclass

from src.modules.ingestion.infra.configs import ByTitleChunkingConfig


@dataclass(frozen=True, slots=True)
class UnstructuredPartitionConfig:
    pdf_strategy: str


@dataclass(frozen=True, slots=True)
class UnstructuredProcessorRuntimeConfig:
    chunking: ByTitleChunkingConfig
    partition: UnstructuredPartitionConfig
    extraction_timeout_seconds: float