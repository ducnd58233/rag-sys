from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class IGraphDb(Protocol):
    async def initialize(self) -> None: ...

    async def write(
        self,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> None: ...

    async def read(
        self,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> Sequence[Mapping[str, object]]: ...

    async def close(self) -> None: ...
