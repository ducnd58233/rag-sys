from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.modules.ingestion.domain.models import (
    Chunk,
    DocumentId,
    DocumentSource,
    ExtractionResult,
)


class ISourceResolver(Protocol):
    async def resolve(
        self,
        source_path: str,
        document_id: DocumentId | None = None,
    ) -> DocumentSource: ...


class IDocumentExtractor(Protocol):
    async def extract(self, source: DocumentSource) -> ExtractionResult: ...


class IChunkingStrategy(Protocol):
    def chunk(self, extraction: ExtractionResult) -> list[Chunk]: ...


class IVectorStore(Protocol):
    async def create_index_if_not_exists(self) -> None: ...
    async def delete_by_document_id(self, document_id: DocumentId) -> None: ...
    async def upsert(
        self,
        document_id: DocumentId,
        metadata: dict[str, str],
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...