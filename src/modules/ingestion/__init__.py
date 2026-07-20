from __future__ import annotations

from dataclasses import dataclass

from src.modules.document.app.ports import IDocumentUnitOfWork
from src.modules.ingestion.app.graph_extraction import LlmDocumentGraphExtractor
from src.modules.ingestion.app.handlers.ingestion_requested import (
    IngestionRequestedHandler,
)
from src.modules.ingestion.app.ports import IDocumentGraphStore, IVectorStore
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
from src.shared.app.ports import (
    IChatModel,
    IEmbeddingModel,
    IIdGenerator,
    IObjectStorage,
)
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
    vector_store: IVectorStore
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
        chat_model: IChatModel,
        id_generator: IIdGenerator,
        graph_store: IDocumentGraphStore,
    ) -> IngestionComponents:
        ingestion = settings.ingestion
        vector_store = ElasticsearchVectorStore(
            elasticsearch=elasticsearch,
            elasticsearch_settings=settings.elasticsearch,
            embedding_settings=settings.embedding,
        )

        ingest_document = IngestDocumentUseCase(
            doc_uow=document_uow,
            source_resolver=ObjectStorageSourceResolver(
                storage=object_storage,
                max_file_size_bytes=(ingestion.max_file_size_bytes),
            ),
            processor=DocumentProcessorFactory.create(ingestion),
            embedder=embedder,
            vector_store=vector_store,
            graph_extractor=LlmDocumentGraphExtractor(
                chat_model,
                settings.graphdb.extraction_max_characters,
            ),
            graph_store=graph_store,
        )

        return IngestionComponents(
            vector_store=vector_store,
            ingest_document=ingest_document,
            request_ingestion=RequestIngestionUseCase(
                doc_uow=document_uow,
                id_generator=id_generator,
            ),
            ingestion_handler=IngestionRequestedHandler(ingest_document),
        )
