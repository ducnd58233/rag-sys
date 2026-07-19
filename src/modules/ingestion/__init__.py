from __future__ import annotations

from dataclasses import dataclass

from src.modules.document.app.ports import (
    IDocumentUnitOfWork,
    IIngestionRequestPublisher,
)
from src.modules.ingestion.app.handlers.ingestion_requested import (
    IngestionRequestedHandler,
)
from src.modules.ingestion.app.use_cases.ingest_document import IngestDocumentUseCase
from src.modules.ingestion.app.use_cases.request_ingestion import (
    RequestIngestionUseCase,
)
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
    "IngestDocumentUseCase",
    "IngestionComponentFactory",
    "IngestionComponents",
    "RequestIngestionUseCase",
]


@dataclass(frozen=True, slots=True)
class IngestionComponents:
    ingest_document: IngestDocumentUseCase
    request_ingestion: RequestIngestionUseCase
    ingestion_handler: IngestionRequestedHandler


class IngestionComponentFactory:
    @staticmethod
    def build(
        *,
        document_uow: IDocumentUnitOfWork,
        object_storage: IObjectStorage,
        settings: Settings,
        elasticsearch: Elasticsearch,
        embedder: IEmbeddingModel,
        ingestion_publisher: IIngestionRequestPublisher,
    ) -> IngestionComponents:
        ingestion = settings.ingestion

        ingest_document = IngestDocumentUseCase(
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

        return IngestionComponents(
            ingest_document=ingest_document,
            request_ingestion=RequestIngestionUseCase(
                doc_uow=document_uow,
                ingestion_publisher=ingestion_publisher,
            ),
            ingestion_handler=IngestionRequestedHandler(ingest_document),
        )
