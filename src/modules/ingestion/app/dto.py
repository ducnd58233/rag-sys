from __future__ import annotations

from dataclasses import dataclass

from src.modules.document.domain.models import DocumentProcessingStatus
from src.modules.ingestion.domain.models import IngestionStatus


@dataclass(frozen=True, slots=True)
class IngestDocumentRequest:
    org_id: int
    document_version_id: int


@dataclass(frozen=True, slots=True)
class IngestDocumentResult:
    document_id: int
    document_version_id: int
    version_no: int
    chunk_count: int
    status: IngestionStatus


@dataclass(frozen=True, slots=True)
class RequestIngestionResult:
    document_id: int
    document_version_id: int
    version_no: int
    processing_status: DocumentProcessingStatus
