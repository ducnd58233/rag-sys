from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from src.modules.generation.domain.models import ContextChunk


class IContextRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        org_id: int,
        top_k: int,
        document_id: int | None = None,
        document_version_id: int | None = None,
        as_of: datetime | None = None,
    ) -> Sequence[ContextChunk]: ...
