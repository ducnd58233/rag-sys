from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClaimDecompositionSchema(BaseModel):
    claims: list[str] = Field(
        default_factory=list,
        description="Atomic factual claims extracted from the answer, one per discrete assertion.",
    )


class FaithfulnessJudgment(BaseModel):
    unsupported_claim_indices: list[int] = Field(
        default_factory=list,
        description="0-based indices into the given claims list of claims NOT supported by the given context.",
    )
    reason: str = ""


class RelevancyJudgment(BaseModel):
    score: int = Field(
        ge=0,
        le=4,
        description="0-4: how well the answer addresses the question, independent of correctness.",
    )
    reason: str = ""


class ClaimCorrectnessJudgment(BaseModel):
    generated_claim_labels: list[Literal["correct", "incorrect"]] = Field(
        default_factory=list,
        description="One label per generated claim, aligned by index, judged against reference_claims.",
    )
    covered_reference_indices: list[int] = Field(
        default_factory=list,
        description="0-based indices into reference_claims that the generated claims support.",
    )


class CitationSupportJudgment(BaseModel):
    supported_pair_indices: list[int] = Field(
        default_factory=list,
        description="0-based indices into the given (claim, cited chunk) pairs where the chunk genuinely supports the claim.",
    )
