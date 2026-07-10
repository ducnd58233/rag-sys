from dataclasses import dataclass

from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import Settings
from src.shared.infra.elasticsearch.client import Elasticsearch


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    elasticsearch: Elasticsearch


def build_container(settings: Settings | None = None) -> AppContainer:
    resolved = settings or Settings()
    configure_logging(resolved.logging)

    return AppContainer(
        settings=resolved,
        elasticsearch=Elasticsearch(resolved.elasticsearch),
    )
