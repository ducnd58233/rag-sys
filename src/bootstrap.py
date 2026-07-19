from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.modules.document import (
    CompleteDocumentUploadUseCase,
    CreateDocumentUploadUrlUseCase,
    DocumentComponentFactory,
)
from src.modules.document.infra.unit_of_work import SqlAlchemyDocumentUnitOfWork
from src.modules.generation import AnswerQuestionUseCase, GenerationComponentFactory
from src.modules.ingestion import IngestDocumentUseCase, IngestionComponentFactory
from src.modules.retrieval import RetrievalComponentFactory, RetrieveUseCase
from src.shared.app.ports import IEmbeddingModel, IObjectStorage
from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import Settings
from src.shared.infra.database import Database
from src.shared.infra.elasticsearch.client import Elasticsearch
from src.shared.infra.embedding import EmbeddingModelFactory
from src.shared.infra.id_generator import SnowflakeIdGenerator
from src.shared.infra.object_storage import MinioObjectStorage


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    database: Database
    object_storage: IObjectStorage
    elasticsearch: Elasticsearch
    embedding_model: IEmbeddingModel
    create_document_upload: CreateDocumentUploadUrlUseCase
    complete_document_upload: CompleteDocumentUploadUseCase
    ingest_document: IngestDocumentUseCase
    retrieve: RetrieveUseCase
    answer_question: AnswerQuestionUseCase

    async def shutdown(self) -> None:
        await asyncio.gather(
            self.elasticsearch.close(),
            self.database.close(),
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

    document_components = DocumentComponentFactory.build(
        settings=resolved.document,
        document_uow=document_uow,
        object_storage=object_storage,
        id_generator=id_generator,
    )
    ingest_document = IngestionComponentFactory.build_ingestion_use_case(
        document_uow=document_uow,
        object_storage=object_storage,
        settings=resolved,
        elasticsearch=elasticsearch,
        embedder=embedding_model,
    )
    retrieve = RetrievalComponentFactory.build_retrieve_use_case(
        settings=resolved,
        elasticsearch=elasticsearch,
        embedder=embedding_model,
    )
    answer_question = GenerationComponentFactory.build_answer_question_use_case(
        settings=resolved,
        retrieve_use_case=retrieve,
    )

    return AppContainer(
        settings=resolved,
        database=database,
        object_storage=object_storage,
        elasticsearch=elasticsearch,
        embedding_model=embedding_model,
        create_document_upload=(document_components.create_upload_url),
        complete_document_upload=(document_components.complete_upload),
        ingest_document=ingest_document,
        retrieve=retrieve,
        answer_question=answer_question,
    )
