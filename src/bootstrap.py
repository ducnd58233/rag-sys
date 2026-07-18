import asyncio

from dataclasses import dataclass

from src.modules.generation import AnswerQuestionUseCase, GenerationComponentFactory
from src.modules.ingestion import IngestionComponentFactory, IngestDocumentUseCase
from src.modules.retrieval import RetrievalComponentFactory, RetrieveUseCase
from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import Settings
from src.shared.infra.database import Database
from src.shared.infra.elasticsearch.client import Elasticsearch
from src.shared.infra.embedding import EmbeddingModelFactory
from src.shared.app.ports import IEmbeddingModel


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    database: Database
    elasticsearch: Elasticsearch
    embedding_model: IEmbeddingModel
    ingest_document: IngestDocumentUseCase
    retrieve: RetrieveUseCase
    answer_question: AnswerQuestionUseCase

    async def shutdown(self) -> None:
        await asyncio.gather(
            self.elasticsearch.close(),
            self.database.close(),
        )


def build_container(settings: Settings | None = None) -> AppContainer:
    resolved = settings or Settings()
    configure_logging(resolved.logging)

    database = Database(resolved.database)
    elasticsearch = Elasticsearch(resolved.elasticsearch)
    embedding_model = EmbeddingModelFactory.from_settings(resolved.embedding)
    ingest_document = IngestionComponentFactory.build_ingestion_use_case(
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
        elasticsearch=elasticsearch,
        embedding_model=embedding_model,
        ingest_document=ingest_document,
        retrieve=retrieve,
        answer_question=answer_question,
    )
