from __future__ import annotations

from src.modules.document.app.ports import IDocumentUnitOfWork
from src.modules.ingestion.app.handlers.ingestion_requested import (
    IngestionRequestedHandler,
)
from src.modules.ingestion.app.use_cases.ingest_document import IngestDocumentUseCase
from src.modules.ingestion.infra.persistence.es_vector_store import (
    ElasticsearchVectorStore,
)
from src.modules.ingestion.infra.processors.factory import DocumentProcessorFactory
from src.modules.ingestion.infra.sources.object_storage_resolver import (
    ObjectStorageSourceResolver,
)
from src.shared.app.ports import IEmbeddingModel, IObjectStorage
from src.shared.configs.settings import Settings
from src.shared.infra.elasticsearch.client import Elasticsearch

__all__ = [
    "IngestionComponentFactory",
    "IngestDocumentUseCase",
]


class IngestionComponentFactory:
    @staticmethod
    def build_ingestion_use_case(
        *,
        document_uow: IDocumentUnitOfWork,
        object_storage: IObjectStorage,
        settings: Settings,
        elasticsearch: Elasticsearch,
        embedder: IEmbeddingModel,
    ) -> IngestDocumentUseCase:
        ingestion = settings.ingestion

        return IngestDocumentUseCase(
            doc_uow=document_uow,
            source_resolver=ObjectStorageSourceResolver(
                storage=object_storage,
                max_file_size_bytes=(ingestion.max_file_size_bytes),
            ),
            processor=DocumentProcessorFactory.create(ingestion),
            embedder=embedder,
            vector_store=ElasticsearchVectorStore(
                elasticsearch=elasticsearch,
                elasticsearch_settings=settings.elasticsearch,
                embedding_settings=settings.embedding,
            ),
        )

    @staticmethod
    def build_ingestion_handler(
        ingest_document: IngestDocumentUseCase,
    ) -> IngestionRequestedHandler:
        return IngestionRequestedHandler(ingest_document)
