from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.modules.document import (
    CompleteDocumentUploadUseCase,
    CreateDocumentUploadUrlUseCase,
    DocumentComponentFactory,
)
from src.modules.document.infra.messaging import KafkaIngestionRequestPublisher
from src.modules.document.infra.unit_of_work import SqlAlchemyDocumentUnitOfWork
from src.modules.generation import AnswerQuestionUseCase, GenerationComponentFactory
from src.modules.ingestion import IngestDocumentUseCase, IngestionComponentFactory
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


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    database: Database
    object_storage: IObjectStorage
    elasticsearch: Elasticsearch
    embedding_model: IEmbeddingModel
    kafka_publisher: AioKafkaPublisher
    create_document_upload: CreateDocumentUploadUrlUseCase
    complete_document_upload: CompleteDocumentUploadUseCase
    ingest_document: IngestDocumentUseCase
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
        )


def build_container(
    settings: Settings | None = None,
) -> AppContainer:
    resolved = settings or Settings()
    configure_logging(resolved.logging)

    database = Database(resolved.database)
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
        database=database,
        object_storage=object_storage,
        elasticsearch=elasticsearch,
        embedding_model=embedding_model,
        kafka_publisher=kafka_publisher,
        create_document_upload=(document_components.create_upload_url),
        complete_document_upload=(document_components.complete_upload),
        ingest_document=ingestion_components.ingest_document,
        ingestion_handler=ingestion_components.ingestion_handler,
        retrieve=retrieval_components.retrieve,
        answer_question=generation_components.answer_question,
    )
