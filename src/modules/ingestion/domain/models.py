from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class IngestionStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class DocumentId:
    value: str


@dataclass(frozen=True, slots=True)
class DocumentSource:
    document_id: DocumentId
    filename: str
    mime_type: str
    source_uri: str
    local_path: Path | None = None
    content: bytes | None = None


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    content: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: DocumentId
    index: int
    content: str
    metadata: dict[str, str]

    @classmethod
    def from_draft(
        cls,
        document_id: DocumentId,
        index: int,
        base_metadata: dict[str, str],
        draft: ChunkDraft,
        chunking_strategy: str,
    ) -> Chunk | None:
        content = draft.content.strip()
        if not content:
            return None

        return cls(
            chunk_id=f"{document_id.value}:{index}",
            document_id=document_id,
            index=index,
            content=content,
            metadata={
                **base_metadata,
                **draft.metadata,
                "chunk_index": str(index),
                "chunking_strategy": chunking_strategy,
            },
        )


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    document_id: DocumentId
    metadata: dict[str, str]
    chunks: tuple[Chunk, ...]