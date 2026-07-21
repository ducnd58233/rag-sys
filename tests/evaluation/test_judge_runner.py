from __future__ import annotations

import pytest
from pydantic import BaseModel
from scripts.evaluation.judge.runner import (
    decompose_claims,
    judge_citation_support,
    judge_claim_correctness,
    judge_faithfulness,
    judge_relevancy,
)
from scripts.evaluation.judge.schemas import (
    CitationSupportJudgment,
    ClaimCorrectnessJudgment,
    ClaimDecompositionSchema,
    FaithfulnessJudgment,
    RelevancyJudgment,
)


class FakeChatModel:
    def __init__(self, responses: list[BaseModel | Exception]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, **kwargs: object) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        temperature: float | None = None,
    ) -> BaseModel:
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def bind_tools(self, tools: object) -> "FakeChatModel":  # pragma: no cover
        return self


@pytest.mark.asyncio
async def test_decompose_claims_returns_empty_without_calling_the_judge_for_empty_answer() -> (
    None
):
    chat = FakeChatModel([])

    result = await decompose_claims(chat, "   ")

    assert result.claims == ()
    assert result.judge_error is None
    assert chat.calls == 0


@pytest.mark.asyncio
async def test_decompose_claims_returns_parsed_claims() -> None:
    chat = FakeChatModel(
        [ClaimDecompositionSchema(claims=["claim one.", "claim two."])],
    )

    result = await decompose_claims(chat, "claim one. claim two.")

    assert result.claims == ("claim one.", "claim two.")
    assert result.judge_error is None
    assert chat.calls == 1


@pytest.mark.asyncio
async def test_decompose_claims_retries_twice_then_succeeds() -> None:
    chat = FakeChatModel(
        [
            ValueError("malformed json"),
            ValueError("malformed json again"),
            ClaimDecompositionSchema(claims=["ok."]),
        ],
    )

    result = await decompose_claims(chat, "some answer")

    assert result.claims == ("ok.",)
    assert result.judge_error is None
    assert chat.calls == 3


@pytest.mark.asyncio
async def test_decompose_claims_records_judge_error_after_exhausting_retries() -> None:
    chat = FakeChatModel(
        [ValueError("bad-1"), ValueError("bad-2"), ValueError("bad-3")],
    )

    result = await decompose_claims(chat, "some answer")

    assert result.claims == ()
    assert result.judge_error is not None
    assert "bad-3" in result.judge_error
    assert chat.calls == 3


@pytest.mark.asyncio
async def test_judge_faithfulness_skips_the_call_for_no_claims() -> None:
    chat = FakeChatModel([])

    result = await judge_faithfulness(chat, claims=(), context="some context")

    assert result.unsupported_claim_indices == ()
    assert result.judge_error is None
    assert chat.calls == 0


@pytest.mark.asyncio
async def test_judge_faithfulness_returns_unsupported_indices_and_reason() -> None:
    chat = FakeChatModel(
        [
            FaithfulnessJudgment(
                unsupported_claim_indices=[1], reason="claim 1 unsupported"
            )
        ],
    )

    result = await judge_faithfulness(
        chat,
        claims=("supported claim", "unsupported claim"),
        context="only supports the first claim",
    )

    assert result.unsupported_claim_indices == (1,)
    assert result.reason == "claim 1 unsupported"
    assert result.judge_error is None


@pytest.mark.asyncio
async def test_judge_relevancy_skips_the_call_for_empty_answer() -> None:
    chat = FakeChatModel([])

    result = await judge_relevancy(chat, question="q", answer="")

    assert result.score is None
    assert chat.calls == 0


@pytest.mark.asyncio
async def test_judge_relevancy_returns_score_and_reason() -> None:
    chat = FakeChatModel([RelevancyJudgment(score=3, reason="mostly on point")])

    result = await judge_relevancy(chat, question="q", answer="a")

    assert result.score == 3
    assert result.reason == "mostly on point"
    assert result.judge_error is None


@pytest.mark.asyncio
async def test_judge_claim_correctness_skips_the_call_when_both_sides_are_empty() -> (
    None
):
    chat = FakeChatModel([])

    result = await judge_claim_correctness(
        chat,
        generated_claims=(),
        reference_claims=(),
    )

    assert result.generated_claim_labels == ()
    assert result.covered_reference_indices == ()
    assert chat.calls == 0


@pytest.mark.asyncio
async def test_judge_claim_correctness_returns_labels_and_covered_indices() -> None:
    chat = FakeChatModel(
        [
            ClaimCorrectnessJudgment(
                generated_claim_labels=["correct", "incorrect"],
                covered_reference_indices=[0],
            ),
        ],
    )

    result = await judge_claim_correctness(
        chat,
        generated_claims=("right claim", "wrong claim"),
        reference_claims=("the reference fact",),
    )

    assert result.generated_claim_labels == ("correct", "incorrect")
    assert result.covered_reference_indices == (0,)
    assert result.judge_error is None


@pytest.mark.asyncio
async def test_judge_citation_support_skips_the_call_for_no_pairs() -> None:
    chat = FakeChatModel([])

    result = await judge_citation_support(chat, ())

    assert result.supported_pair_indices == ()
    assert chat.calls == 0


@pytest.mark.asyncio
async def test_judge_citation_support_returns_supported_pair_indices() -> None:
    chat = FakeChatModel([CitationSupportJudgment(supported_pair_indices=[0])])

    result = await judge_citation_support(
        chat,
        (("claim", "passage that supports it"), ("claim2", "unrelated passage")),
    )

    assert result.supported_pair_indices == (0,)
    assert result.judge_error is None
