from __future__ import annotations

from pydantic import BaseModel, Field


class GroundedAnswerSchema(BaseModel):
    refused: bool = Field(
        description="True when context does not answer the question."
    )
    answer: str = Field(description="Final answer text shown to the user.")
    cited_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices of supporting context chunks.",
    )