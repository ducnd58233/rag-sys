from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel
from scripts.evaluation.judge import rubrics
from scripts.evaluation.judge.config import JUDGE_TEMPERATURE
from scripts.evaluation.judge.schemas import (
    CitationSupportJudgment,
    ClaimCorrectnessJudgment,
    ClaimDecompositionSchema,
    FaithfulnessJudgment,
    RelevancyJudgment,
)

from src.shared.app.ports import IChatModel

TSchema = TypeVar("TSchema", bound=BaseModel)

_MAX_ATTEMPTS = 3  # first attempt + two retries, per FR-EVAL-6


async def _call_with_retry(
    chat: IChatModel,
    *,
    system: str,
    user: str,
    schema: type[TSchema],
    temperature: float = JUDGE_TEMPERATURE,
) -> tuple[TSchema | None, str | None]:
    last_error: Exception | None = None
    for _attempt in range(_MAX_ATTEMPTS):
        try:
            result = await chat.complete_structured(
                system=system,
                user=user,
                schema=schema,
                temperature=temperature,
            )
            return result, None
        except (
            Exception
        ) as error:  # noqa: BLE001 - any malformed/unparseable judge output
            last_error = error
    return None, f"{type(last_error).__name__}: {last_error}"


@dataclass(frozen=True, slots=True)
class ClaimsResult:
    claims: tuple[str, ...]
    judge_error: str | None


async def decompose_claims(chat: IChatModel, answer: str) -> ClaimsResult:
    if not answer.strip():
        return ClaimsResult(claims=(), judge_error=None)
    parsed, error = await _call_with_retry(
        chat,
        system=rubrics.CLAIM_DECOMPOSITION_SYSTEM,
        user=rubrics.claim_decomposition_user(answer),
        schema=ClaimDecompositionSchema,
    )
    if error is not None or parsed is None:
        return ClaimsResult(claims=(), judge_error=error)
    return ClaimsResult(claims=tuple(parsed.claims), judge_error=None)


@dataclass(frozen=True, slots=True)
class FaithfulnessResult:
    unsupported_claim_indices: tuple[int, ...]
    reason: str
    judge_error: str | None


async def judge_faithfulness(
    chat: IChatModel,
    *,
    claims: tuple[str, ...],
    context: str,
) -> FaithfulnessResult:
    if not claims:
        return FaithfulnessResult((), "", None)
    parsed, error = await _call_with_retry(
        chat,
        system=rubrics.FAITHFULNESS_SYSTEM,
        user=rubrics.faithfulness_user(claims=list(claims), context=context),
        schema=FaithfulnessJudgment,
    )
    if error is not None or parsed is None:
        return FaithfulnessResult((), "", error)
    return FaithfulnessResult(
        tuple(parsed.unsupported_claim_indices),
        parsed.reason,
        None,
    )


@dataclass(frozen=True, slots=True)
class RelevancyResult:
    score: int | None
    reason: str
    judge_error: str | None


async def judge_relevancy(
    chat: IChatModel,
    *,
    question: str,
    answer: str,
) -> RelevancyResult:
    if not answer.strip():
        return RelevancyResult(score=None, reason="", judge_error=None)
    parsed, error = await _call_with_retry(
        chat,
        system=rubrics.RELEVANCY_SYSTEM,
        user=rubrics.relevancy_user(question=question, answer=answer),
        schema=RelevancyJudgment,
    )
    if error is not None or parsed is None:
        return RelevancyResult(score=None, reason="", judge_error=error)
    return RelevancyResult(parsed.score, parsed.reason, None)


@dataclass(frozen=True, slots=True)
class ClaimCorrectnessResult:
    generated_claim_labels: tuple[str, ...]
    covered_reference_indices: tuple[int, ...]
    judge_error: str | None


async def judge_claim_correctness(
    chat: IChatModel,
    *,
    generated_claims: tuple[str, ...],
    reference_claims: tuple[str, ...],
) -> ClaimCorrectnessResult:
    if not generated_claims and not reference_claims:
        return ClaimCorrectnessResult((), (), None)
    parsed, error = await _call_with_retry(
        chat,
        system=rubrics.CLAIM_CORRECTNESS_SYSTEM,
        user=rubrics.claim_correctness_user(
            generated_claims=list(generated_claims),
            reference_claims=list(reference_claims),
        ),
        schema=ClaimCorrectnessJudgment,
    )
    if error is not None or parsed is None:
        return ClaimCorrectnessResult((), (), error)
    return ClaimCorrectnessResult(
        tuple(parsed.generated_claim_labels),
        tuple(parsed.covered_reference_indices),
        None,
    )


@dataclass(frozen=True, slots=True)
class CitationSupportResult:
    supported_pair_indices: tuple[int, ...]
    judge_error: str | None


async def judge_citation_support(
    chat: IChatModel,
    pairs: tuple[tuple[str, str], ...],
) -> CitationSupportResult:
    if not pairs:
        return CitationSupportResult((), None)
    parsed, error = await _call_with_retry(
        chat,
        system=rubrics.CITATION_SUPPORT_SYSTEM,
        user=rubrics.citation_support_user(list(pairs)),
        schema=CitationSupportJudgment,
    )
    if error is not None or parsed is None:
        return CitationSupportResult((), error)
    return CitationSupportResult(tuple(parsed.supported_pair_indices), None)
