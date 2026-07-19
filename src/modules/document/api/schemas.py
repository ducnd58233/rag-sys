from __future__ import annotations

from pydantic import BaseModel, Field


class CreateDocumentRequestBody(BaseModel):
    org_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)


class DocumentUploadResponse(BaseModel):
    document_id: int
    document_version_id: int
    version_no: int
    signed_put_url: str
    expires_in_seconds: int


class CompleteDocumentVersionRequestBody(BaseModel):
    org_id: int = Field(gt=0)
    checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-fA-F]{64}$",
    )


class CompleteDocumentVersionResponse(BaseModel):
    document_id: int
    document_version_id: int
    version_no: int
