from __future__ import annotations
from dataclasses import dataclass
from src.modules.ingestion.app.ports import (
    IChunkingStrategy,
    IDocumentExtractor,
    ISourceResolver,
    IVectorStore,
)
from src.modules.ingestion.app.use_cases.ingest_document import IngestDocumentUseCase
from src.modules.ingestion.infra.chunkers.unstructured_by_title_strategy import (
    UnstructuredByTitleStrategy,
)
from src.modules.ingestion.infra.configs import ByTitleChunkingConfig, SourceResolverConfig
from src.modules.ingestion.infra.extractors.unstructured_extractor import (
    UnstructuredExtractor,
)
from src.modules.ingestion.infra.persistence.es_vector_store import (
    ElasticsearchVectorStore,
)
from src.modules.ingestion.infra.sources.file_source_resolver import FileSourceResolver
from src.shared.configs.settings import Settings
from src.shared.infra.elasticsearch.client import Elasticsearch
from src.shared.infra.embedding.ports import IEmbeddingModel

__all__ = ["IngestionComponentFactory",  "IngestDocumentUseCase"]

@dataclass(frozen=True, slots=True)
class IngestionComponents:
    source_resolver: ISourceResolver
    extractor: IDocumentExtractor
    chunking_strategy: IChunkingStrategy
    vector_store: IVectorStore

class IngestionComponentFactory:
    @staticmethod
    def build_ingestion_use_case(
        settings: Settings,
        elasticsearch: Elasticsearch,
        embedder: IEmbeddingModel,
    ) -> IngestDocumentUseCase:
        ingestion = settings.ingestion
        source_resolver = FileSourceResolver(
            SourceResolverConfig(
                max_file_size_bytes=ingestion.max_file_size_bytes,
            ),
        )
        chunking_strategy = UnstructuredByTitleStrategy(
            ByTitleChunkingConfig(
                max_characters=ingestion.chunk_max_characters,
                combine_text_under_n_chars=ingestion.chunk_combine_text_under_n_chars,
                new_after_n_chars=ingestion.chunk_new_after_n_chars,
            ),
        )
        return IngestDocumentUseCase(
            source_resolver=source_resolver,
            extractor=UnstructuredExtractor(),
            chunking_strategy=chunking_strategy,
            embedder=embedder,
            vector_store=ElasticsearchVectorStore(
                elasticsearch=elasticsearch,
                elasticsearch_settings=settings.elasticsearch,
                embedding_settings=settings.embedding,
            ),
        )
