from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from src.modules.ingestion.domain.element_category import ElementCategory


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
class ExtractedElement:
    text: str
    category: ElementCategory
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    document_id: DocumentId
    elements: tuple[ExtractedElement, ...]
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: DocumentId
    index: int
    content: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: DocumentId
    chunk_count: int
    status: IngestionStatus
    indexed_at: datetime