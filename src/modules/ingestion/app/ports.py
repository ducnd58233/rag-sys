from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from src.modules.ingestion.domain.models import (
    Chunk,
    DocumentSource,
    DocumentVersionSource,
    ProcessedDocument,
)
from src.modules.ingestion.domain.models import DocumentId


class ISourceResolver(Protocol):
    def open(
        self,
        source: DocumentVersionSource,
    ) -> AbstractAsyncContextManager[DocumentSource]: ...


class IDocumentProcessor(Protocol):
    async def process(
        self,
        source: DocumentSource,
    ) -> ProcessedDocument: ...


class IVectorStore(Protocol):
    async def create_index_if_not_exists(self) -> None: ...

    async def delete_by_document_id(
        self,
        document_id: DocumentId,
    ) -> None: ...

    async def upsert(
        self,
        document_id: DocumentId,
        metadata: dict[str, str],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...