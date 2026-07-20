from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RetrieveRequestBody(BaseModel):
    org_id: int = Field(gt=0)
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    document_id: int | None = Field(default=None, gt=0)
    document_version_id: int | None = Field(default=None, gt=0)
    as_of: datetime | None = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, str]


class StrategySelectionResponse(BaseModel):
    name: str
    weight: float
    top_k: int


class RetrievalPlanResponse(BaseModel):
    strategies: list[StrategySelectionResponse]
    router_kind: str
    reason: str
    confidence: float


class RetrieveResponse(BaseModel):
    query: str
    plan: RetrievalPlanResponse
    items: list[RetrievedChunkResponse]
