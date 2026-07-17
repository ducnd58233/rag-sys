from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveRequestBody(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    document_id: str | None = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, str]


class RetrieveResponse(BaseModel):
    query: str
    items: list[RetrievedChunkResponse]