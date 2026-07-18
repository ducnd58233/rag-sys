from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum

class DocumentStatus(StrEnum):
    ACTIVE = "active"
    DELETED = "deleted"

class DocumentProcessingStatus(StrEnum):
    UPLOADED = "uploaded"
    SCANNING = "scanning"
    PARSING = "parsing"
    INDEXED = "indexed"
    FAILED = "failed"
    QUARANTINED = "quarantined"

class DocumentScanStatus(StrEnum):
    PENDING = "pending"
    CLEAN = "clean"
    INFECTED = "infected"
    FAILED = "failed"

class StoredObjectStatus(StrEnum):
    PENDING = 'pending'
    AVAILABLE = 'available'
    QUARANTINED = 'quarantined'
    DELETED = 'deleted'
    FAILED = 'failed'
    
@dataclass(frozen=True, slots=True)
class CaseRecord:
    id: int
    org_id: int
    
@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: int
    org_id: int
    case_id: int
    display_name: str
    current_version_id: int | None
    status: DocumentStatus
    created_by: int

@dataclass(frozen=True, slots=True)
class DocumentVersionRecord:
    id: int
    org_id: int
    document_id: int
    storage_object_id: int
    version_no: int
    filename: str
    mime_type: str
    processing_status: DocumentProcessingStatus
    scan_status: DocumentScanStatus
    uploaded_by: int
    doc_type: str | None
    doc_type_confidence: float | None

@dataclass(frozen=True, slots=True)
class StoredObjectRecord:
    id: int
    org_id: int
    bucket: str
    object_key: str
    purpose: str
    content_type: str
    size_bytes: int
    checksum_sha256: str | None
    etag: str | None
    status: StoredObjectStatus
    created_by: int | None
