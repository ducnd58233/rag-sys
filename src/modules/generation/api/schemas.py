from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequestBody(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    document_id: str | None = None


class CitationResponse(BaseModel):
    chunk_id: str
    document_id: str


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationResponse]
    refused: bool