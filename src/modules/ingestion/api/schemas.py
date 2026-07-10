from __future__ import annotations

from pydantic import BaseModel, Field

class IngestDocumentResponse(BaseModel):
    document_id: str
    chunk_count: int
    status: str
    
class IngestByPathRequest(BaseModel):
    source_path: str = Field(min_length=1)
    document_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)