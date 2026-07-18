from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.modules.document.domain.models import DocumentProcessingStatus, DocumentScanStatus


class IngestionStatus(StrEnum):
    COMPLETED = 'completed'
    FAILED = 'failed'


@dataclass(frozen=True, slots=True)
class DocumentId:
    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError('document id must be positive')


@dataclass(frozen=True, slots=True)
class DocumentSource:
    document_id: DocumentId
    version_no: int
    filename: str
    mime_type: str
    source_uri: str
    local_path: Path | None = None
    content: bytes | None = None

@dataclass(frozen=True, slots=True)
class DocumentVersionSource:
    org_id: int
    case_id: int
    document_id: DocumentId
    document_version_id: int
    storage_object_id: int
    version_no: int
    bucket: str
    object_key: str
    filename: str
    mime_type: str
    processing_status: DocumentProcessingStatus
    scan_status: DocumentScanStatus
    size_bytes: int
    checksum_sha256: str | None
    def __post_init__(self) -> None:
        positive_values = (
            self.org_id,
            self.case_id,
            self.document_version_id,
            self.storage_object_id,
            self.version_no,
            self.size_bytes,
        )
        if any(value < 0 for value in positive_values):
            raise ValueError('document version fields must be non-negative')
        if any(
            value <= 0
            for value in (
                self.org_id,
                self.case_id,
                self.document_version_id,
                self.storage_object_id,
                self.version_no,
            )
        ):
            raise ValueError('document version identifiers must be positive')


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
        version_no: int,
        index: int,
        base_metadata: dict[str, str],
        draft: ChunkDraft,
        chunking_strategy: str,
    ) -> Chunk | None:
        content = draft.content.strip()
        if not content:
            return None

        return cls(
            chunk_id=f'{document_id.value}:{version_no}:{index}',
            document_id=document_id,
            index=index,
            content=content,
            metadata={
                **base_metadata,
                **draft.metadata,
                'version': str(version_no),
                'chunk_index': str(index),
                'chunking_strategy': chunking_strategy,
            },
        )


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    document_id: DocumentId
    metadata: dict[str, str]
    chunks: tuple[Chunk, ...]