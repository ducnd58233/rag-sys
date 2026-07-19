from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from src.modules.ingestion.domain.models import (
    Chunk,
    DocumentId,
    DocumentSource,
    DocumentVersionSource,
    ProcessedDocument,
)


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
        *,
        org_id: int,
        document_id: DocumentId,
    ) -> None: ...

    async def upsert(
        self,
        *,
        org_id: int,
        document_id: DocumentId,
        document_version_id: int,
        version_no: int,
        metadata: dict[str, str],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...
