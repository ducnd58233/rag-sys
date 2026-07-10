from __future__ import annotations

from dataclasses import dataclass, field

from src.modules.ingestion.domain.models import IngestionStatus


@dataclass(frozen=True, slots=True)
class IngestDocumentRequest:
    source_path: str
    document_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestDocumentResult:
    document_id: str
    chunk_count: int
    status: IngestionStatus