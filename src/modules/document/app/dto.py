from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CreateUploadUrlRequest:
    org_id: int
    user_id: int
    filename: str
    mime_type: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class CreateUploadUrlResult:
    document_id: int
    document_version_id: int
    version_no: int
    upload_url: str
    expires_in_seconds: int


@dataclass(frozen=True, slots=True)
class CompleteUploadRequest:
    org_id: int
    document_version_id: int
    checksum_sha256: str


@dataclass(frozen=True, slots=True)
class CompleteUploadResult:
    document_id: int
    document_version_id: int
    version_no: int
