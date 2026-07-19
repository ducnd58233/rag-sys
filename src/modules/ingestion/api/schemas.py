from __future__ import annotations

from pydantic import BaseModel, Field


class CreateIngestionRequestBody(BaseModel):
    org_id: int = Field(gt=0)
    document_version_id: int = Field(gt=0)


class IngestionAcceptedResponse(BaseModel):
    document_id: int
    document_version_id: int
    version_no: int
    status: str
