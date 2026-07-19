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
    covered_questions: list[int] = Field(
        default_factory=list,
        description="1-based answer-intent indices covered by the answer.",
    )
    unsupported_questions: list[int] = Field(
        default_factory=list,
        description="1-based answer-intent indices not supported by context.",
    )


class QueryAnalysisSchema(BaseModel):
    is_complex: bool = Field(
        description="True when the query has multiple distinct answer intents."
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Self-contained sub-questions for retrieval.",
    )
    reason: str = Field(
        default="",
        description="Short reason for the classification.",
    )
