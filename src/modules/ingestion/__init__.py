from __future__ import annotations

from dataclasses import dataclass

from src.modules.document.infra.unit_of_work import SqlAlchemyDocumentUnitOfWork
from src.modules.ingestion.app.ports import (
    IDocumentProcessor,
    ISourceResolver,
    IVectorStore,
)
from src.modules.ingestion.app.use_cases.ingest_document import IngestDocumentUseCase
from src.modules.ingestion.infra.persistence.es_vector_store import ElasticsearchVectorStore
from src.modules.ingestion.infra.processors.factory import DocumentProcessorFactory
from src.modules.ingestion.infra.sources.object_storage_resolver import ObjectStorageSourceResolver
from src.shared.configs.settings import Settings
from src.shared.infra.database import Database
from src.shared.infra.elasticsearch.client import Elasticsearch
from src.shared.app.ports import IEmbeddingModel

__all__ = ["IngestionComponentFactory", "IngestDocumentUseCase"]


@dataclass(frozen=True, slots=True)
class IngestionComponents:
    source_resolver: ISourceResolver
    processor: IDocumentProcessor
    vector_store: IVectorStore


class IngestionComponentFactory:
    @staticmethod
    def build_ingestion_use_case(
        *,
        database: Database,
        settings: Settings,
        elasticsearch: Elasticsearch,
        embedder: IEmbeddingModel,
    ) -> IngestDocumentUseCase:
        ingestion = settings.ingestion
        doc_uow = SqlAlchemyDocumentUnitOfWork(database.session_factory)
        source_resolver = ObjectStorageSourceResolver(
            storage=settings.object_storage,
            max_file_size_bytes=ingestion.max_file_size_bytes,
        )
        processor = DocumentProcessorFactory.create(ingestion)
        vector_store = ElasticsearchVectorStore(
            elasticsearch=elasticsearch,
            elasticsearch_settings=settings.elasticsearch,
            embedding_settings=settings.embedding,
        )
        return IngestDocumentUseCase(
            doc_uow=doc_uow,
            source_resolver=source_resolver,
            processor=processor,
            embedder=embedder,
            vector_store=vector_store,
        )