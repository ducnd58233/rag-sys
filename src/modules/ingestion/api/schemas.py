from __future__ import annotations

from pydantic import BaseModel, Field


class CreateIngestionRequestBody(BaseModel):
    org_id: int = Field(gt=0)
    document_version_id: int = Field(gt=0)


class IngestDocumentResponse(BaseModel):
    document_id: int
    document_version_id: int
    version_no: int
    chunk_count: int
    status: str
