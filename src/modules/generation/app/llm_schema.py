from __future__ import annotations

from pydantic import BaseModel, Field


class GroundedAnswerSchema(BaseModel):
    refused: bool = Field(description="True when context does not answer the question.")
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


class MathEvidenceSchema(BaseModel):
    has_math_evidence: bool = Field(
        description="True when formula_latex is visibly supported by context.",
    )
    formula_latex: str = Field(
        default="",
        description="Verified LaTeX formula evidence, or empty when unsupported.",
    )


class QueryIntentSchema(BaseModel):
    question: str = Field(
        description="Self-contained user intent question for answer synthesis.",
    )
    retrieval_queries: list[str] = Field(
        default_factory=list,
        description="Concise search queries for this intent.",
    )


class QueryAnalysisSchema(BaseModel):
    is_complex: bool = Field(
        description="True when the query has multiple distinct answer intents."
    )
    rewritten_query: str = Field(
        default="",
        description="A clearer standalone version of the user query.",
    )
    intents: list[QueryIntentSchema] = Field(
        default_factory=list,
        description="Answer intents with optimized retrieval queries.",
    )
    reason: str = Field(
        default="",
        description="Short reason for the classification.",
    )
