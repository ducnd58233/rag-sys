from dataclasses import dataclass

from shared.configs.logger import configure_logging
from shared.configs.settings import Settings


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings


def build_container(settings: Settings | None = None) -> AppContainer:
    resolved = settings or Settings()
    configure_logging(resolved.logging)

    return AppContainer(
        settings=resolved,
    )
