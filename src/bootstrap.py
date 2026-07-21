from __future__ import annotations

import asyncio
import logging
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
from src.modules.ingestion.app.ports import IVectorStore
from src.modules.ingestion.infra.graphdb import Neo4jDocumentGraphStore
from src.modules.retrieval import RetrievalComponentFactory, RetrieveUseCase
from src.modules.retrieval.infra.graphdb import Neo4jGraphSearcher
from src.shared.app.ports import IChatModel, IEmbeddingModel, IGraphDb, IObjectStorage
from src.shared.app.retry import RetryPolicy
from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import Settings
from src.shared.infra.chat import ChatModelFactory
from src.shared.infra.database import Database
from src.shared.infra.elasticsearch.client import Elasticsearch
from src.shared.infra.embedding import EmbeddingModelFactory
from src.shared.infra.graphdb import Neo4jGraphDb
from src.shared.infra.id_generator import SnowflakeIdGenerator
from src.shared.infra.mq import AioKafkaPublisher
from src.shared.infra.object_storage import MinioObjectStorage
from src.shared.observability import Observability, setup_observability

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    observability: Observability
    database: Database
    object_storage: IObjectStorage
    elasticsearch: Elasticsearch
    graphdb: IGraphDb
    embedding_model: IEmbeddingModel
    chat_model: IChatModel
    kafka_publisher: AioKafkaPublisher
    create_document_upload: CreateDocumentUploadUrlUseCase
    complete_document_upload: CompleteDocumentUploadUseCase
    publish_ingestion_outbox_events: PublishIngestionOutboxEventsUseCase
    vector_store: IVectorStore
    ingest_document: IngestDocumentUseCase
    request_ingestion: RequestIngestionUseCase
    ingestion_handler: IngestionRequestedHandler
    retrieve: RetrieveUseCase
    answer_question: AnswerQuestionUseCase

    async def startup(self) -> None:
        await self.graphdb.initialize()
        await self.vector_store.create_index_if_not_exists()
        await self.kafka_publisher.start()
        await self._warmup()

    async def _warmup(self) -> None:
        resilience = self.settings.resilience
        if not resilience.warmup_enabled:
            return

        try:
            if resilience.warmup_embedding:
                await self.embedding_model.embed([resilience.warmup_embedding_text])
            if resilience.warmup_chat:
                await self.chat_model.complete(
                    system=resilience.warmup_chat_system,
                    user=resilience.warmup_chat_user,
                )
        except Exception:
            if resilience.warmup_fail_fast:
                raise
            logger.exception("Infra warmup failed; continuing startup")

    async def shutdown(self) -> None:
        await asyncio.gather(
            self.elasticsearch.close(),
            self.graphdb.close(),
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
    graphdb = Neo4jGraphDb(resolved.graphdb)

    retry_policy = RetryPolicy(
        max_attempts=resolved.resilience.retry_max_attempts,
        base_delay_seconds=resolved.resilience.retry_base_delay_seconds,
        max_delay_seconds=resolved.resilience.retry_max_delay_seconds,
    )
    embedding_model = EmbeddingModelFactory.from_settings(
        resolved.embedding,
        retry_policy=retry_policy,
    )
    chat_model = ChatModelFactory.from_settings(
        resolved.chat,
        retry_policy=retry_policy,
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
        chat_model=chat_model,
        id_generator=id_generator,
        graph_store=Neo4jDocumentGraphStore(graphdb),
    )
    retrieval_components = RetrievalComponentFactory.build(
        settings=resolved,
        elasticsearch=elasticsearch,
        embedder=embedding_model,
        chat_model=chat_model,
        graph_searcher=Neo4jGraphSearcher(graphdb),
    )
    generation_components = GenerationComponentFactory.build(
        settings=resolved,
        retrieve_use_case=retrieval_components.retrieve,
        chat_model=chat_model,
    )

    return AppContainer(
        settings=resolved,
        observability=observability,
        database=database,
        object_storage=object_storage,
        elasticsearch=elasticsearch,
        graphdb=graphdb,
        embedding_model=embedding_model,
        chat_model=chat_model,
        kafka_publisher=kafka_publisher,
        create_document_upload=(document_components.create_upload_url),
        complete_document_upload=(document_components.complete_upload),
        publish_ingestion_outbox_events=(
            document_components.publish_ingestion_outbox_events
        ),
        vector_store=ingestion_components.vector_store,
        ingest_document=ingestion_components.ingest_document,
        request_ingestion=ingestion_components.request_ingestion,
        ingestion_handler=ingestion_components.ingestion_handler,
        retrieve=retrieval_components.retrieve,
        answer_question=generation_components.answer_question,
    )
