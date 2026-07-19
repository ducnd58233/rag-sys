from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequestBody(BaseModel):
    org_id: int = Field(gt=0)
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    document_id: int | None = Field(default=None, gt=0)
    document_version_id: int | None = Field(default=None, gt=0)


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationResponse]
    refused: bool
