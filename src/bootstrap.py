from __future__ import annotations

import asyncio
from dataclasses import dataclass

from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from src.modules.document import (
    CompleteDocumentUploadUseCase,
    CreateDocumentUploadUrlUseCase,
    DocumentComponentFactory,
    PublishIngestionOutboxEventsUseCase,
)
from src.modules.document.infra.messaging import KafkaIngestionRequestPublisher
from src.modules.document.infra.unit_of_work import SqlAlchemyDocumentUnitOfWork
from src.modules.generation import AnswerQuestionUseCase, GenerationComponentFactory
from src.modules.ingestion import (
    IngestDocumentUseCase,
    IngestionComponentFactory,
    RequestIngestionUseCase,
)
from src.modules.ingestion.app.handlers.ingestion_requested import (
    IngestionRequestedHandler,
)
from src.modules.retrieval import RetrievalComponentFactory, RetrieveUseCase
from src.shared.app.ports import IEmbeddingModel, IObjectStorage
from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import Settings
from src.shared.infra.database import Database
from src.shared.infra.elasticsearch.client import Elasticsearch
from src.shared.infra.embedding import EmbeddingModelFactory
from src.shared.infra.id_generator import SnowflakeIdGenerator
from src.shared.infra.mq import AioKafkaPublisher
from src.shared.infra.object_storage import MinioObjectStorage
from src.shared.observability import Observability, setup_observability


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    observability: Observability
    database: Database
    object_storage: IObjectStorage
    elasticsearch: Elasticsearch
    embedding_model: IEmbeddingModel
    kafka_publisher: AioKafkaPublisher
    create_document_upload: CreateDocumentUploadUrlUseCase
    complete_document_upload: CompleteDocumentUploadUseCase
    publish_ingestion_outbox_events: PublishIngestionOutboxEventsUseCase
    ingest_document: IngestDocumentUseCase
    request_ingestion: RequestIngestionUseCase
    ingestion_handler: IngestionRequestedHandler
    retrieve: RetrieveUseCase
    answer_question: AnswerQuestionUseCase

    async def startup(self) -> None:
        await self.kafka_publisher.start()

    async def shutdown(self) -> None:
        await asyncio.gather(
            self.elasticsearch.close(),
            self.database.close(),
            self.kafka_publisher.close(),
            self.observability.shutdown(),
        )


def build_container(
    settings: Settings | None = None,
    *,
    service_name: str = "api",
) -> AppContainer:
    resolved = settings or Settings()
    observability = setup_observability(
        resolved.observability,
        service_name=service_name,
    )
    configure_logging(resolved.logging, service_name=service_name)

    database = Database(resolved.database)
    if resolved.observability.enabled:
        SQLAlchemyInstrumentor().instrument(
            engine=database.engine.sync_engine,
            enable_commenter=False,
        )
    object_storage = MinioObjectStorage(
        resolved.object_storage,
    )
    elasticsearch = Elasticsearch(resolved.elasticsearch)
    embedding_model = EmbeddingModelFactory.from_settings(
        resolved.embedding,
    )
    id_generator = SnowflakeIdGenerator(
        resolved.snowflake.instance_id,
    )
    document_uow = SqlAlchemyDocumentUnitOfWork(
        database.session_factory,
    )
    kafka_publisher = AioKafkaPublisher(resolved.kafka)
    ingestion_publisher = KafkaIngestionRequestPublisher(kafka_publisher)

    document_components = DocumentComponentFactory.build(
        settings=resolved.document,
        document_uow=document_uow,
        object_storage=object_storage,
        id_generator=id_generator,
        ingestion_publisher=ingestion_publisher,
    )
    ingestion_components = IngestionComponentFactory.build(
        document_uow=document_uow,
        object_storage=object_storage,
        settings=resolved,
        elasticsearch=elasticsearch,
        embedder=embedding_model,
        id_generator=id_generator,
    )
    retrieval_components = RetrievalComponentFactory.build(
        settings=resolved,
        elasticsearch=elasticsearch,
        embedder=embedding_model,
    )
    generation_components = GenerationComponentFactory.build(
        settings=resolved,
        retrieve_use_case=retrieval_components.retrieve,
    )

    return AppContainer(
        settings=resolved,
        observability=observability,
        database=database,
        object_storage=object_storage,
        elasticsearch=elasticsearch,
        embedding_model=embedding_model,
        kafka_publisher=kafka_publisher,
        create_document_upload=(document_components.create_upload_url),
        complete_document_upload=(document_components.complete_upload),
        publish_ingestion_outbox_events=(
            document_components.publish_ingestion_outbox_events
        ),
        ingest_document=ingestion_components.ingest_document,
        request_ingestion=ingestion_components.request_ingestion,
        ingestion_handler=ingestion_components.ingestion_handler,
        retrieve=retrieval_components.retrieve,
        answer_question=generation_components.answer_question,
    )
