from dataclasses import dataclass

from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import Settings
from src.shared.infra.elasticsearch.client import Elasticsearch
from src.shared.infra.embedding import EmbeddingModelFactory, IEmbeddingModel


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    elasticsearch: Elasticsearch
    embedding_model: IEmbeddingModel


def build_container(settings: Settings | None = None) -> AppContainer:
    resolved = settings or Settings()
    configure_logging(resolved.logging)

    return AppContainer(
        settings=resolved,
        elasticsearch=Elasticsearch(resolved.elasticsearch),
        embedding_model=EmbeddingModelFactory.from_settings(resolved.embedding),
    )
