from collections.abc import Sequence
from typing import Protocol


class IEmbeddingModel(Protocol):
    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...