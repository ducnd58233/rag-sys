from dataclasses import dataclass

from src.modules.ingestion import IngestionComponentFactory, IngestDocumentUseCase
from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import Settings
from src.shared.infra.elasticsearch.client import Elasticsearch
from src.shared.infra.embedding import EmbeddingModelFactory
from src.shared.app.ports import IEmbeddingModel


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    elasticsearch: Elasticsearch
    embedding_model: IEmbeddingModel
    ingest_document: IngestDocumentUseCase

    async def shutdown(self) -> None:
        await self.elasticsearch.close()


def build_container(settings: Settings | None = None) -> AppContainer:
    resolved = settings or Settings()
    configure_logging(resolved.logging)

    elasticsearch = Elasticsearch(resolved.elasticsearch)
    embedding_model = EmbeddingModelFactory.from_settings(resolved.embedding)
    ingest_document = IngestionComponentFactory.build_ingestion_use_case(
        settings=resolved,
        elasticsearch=elasticsearch,
        embedder=embedding_model,
    )

    return AppContainer(
        settings=resolved,
        elasticsearch=elasticsearch,
        embedding_model=embedding_model,
        ingest_document=ingest_document,
    )
