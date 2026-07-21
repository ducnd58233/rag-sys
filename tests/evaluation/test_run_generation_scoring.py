from __future__ import annotations

import pytest
from pydantic import BaseModel
from scripts.evaluation.client import AskCitation, AskResponse
from scripts.evaluation.dataset import EvalCase
from scripts.evaluation.run import _score_generation


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


def _case(
    *, reference_claims: tuple[str, ...] = ("ref claim one", "ref claim two")
) -> EvalCase:
    return EvalCase(
        id="case-1",
        question="What connects A and B?",
        reference_answer="A and B are connected.",
        reference_claims=reference_claims,
        relevant_document_ids=("doc-1",),
        relevance_grades={"doc-1": 1},
        answerable=True,
        difficulty="single-hop",
        split="dev",
        tags=(),
        reviewed_by=None,
    )


def _ask_response(
    *, answer: str, refused: bool, citation_count: int = 2
) -> AskResponse:
    citations = tuple(
        AskCitation(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            content=f"supporting content {i}",
            score=0.9,
        )
        for i in range(citation_count)
    )
    return AskResponse(
        answer=answer, citations=citations, refused=refused, latency_ms=12.0
    )


@pytest.mark.asyncio
async def test_refused_response_skips_every_judge_call() -> None:
    chat = FakeChatModel([])
    ask_response = _ask_response(answer="", refused=True, citation_count=0)

    record = await _score_generation(chat, _case(), ask_response)

    assert record["refused"] is True
    assert record["claims"] == []
    assert record["generated_claim_labels"] == []
    assert record["covered_reference_indices"] == []
    assert record["generation_metrics"] == {
        "faithfulness": None,
        "answer_relevancy": None,
        "claim_precision": None,
        "claim_recall": None,
        "claim_f1": None,
        "completeness": None,
    }
    assert record["citation_metrics"] == {
        "citation_precision": None,
        "citation_recall": None,
    }
    assert chat.calls == 0


@pytest.mark.asyncio
async def test_full_scoring_pipeline_matches_hand_computed_metrics() -> None:
    from scripts.evaluation.judge.schemas import (
        CitationSupportJudgment,
        ClaimCorrectnessJudgment,
        ClaimDecompositionSchema,
        FaithfulnessJudgment,
        RelevancyJudgment,
    )

    chat = FakeChatModel(
        [
            ClaimDecompositionSchema(claims=["Claim A.", "Claim B."]),
            FaithfulnessJudgment(unsupported_claim_indices=[1], reason="B unsupported"),
            RelevancyJudgment(score=4, reason="on point"),
            ClaimCorrectnessJudgment(
                generated_claim_labels=["correct", "incorrect"],
                covered_reference_indices=[0],
            ),
            CitationSupportJudgment(supported_pair_indices=[0]),
        ],
    )
    ask_response = _ask_response(
        answer="Claim A. Claim B.", refused=False, citation_count=2
    )

    record = await _score_generation(chat, _case(), ask_response)

    assert record["claims"] == ["Claim A.", "Claim B."]
    assert record["unsupported_claim_indices"] == [1]
    assert record["generated_claim_labels"] == ["correct", "incorrect"]
    assert record["covered_reference_indices"] == [0]
    # faithfulness: (2 claims - 1 unsupported) / 2 = 0.5
    assert record["generation_metrics"]["faithfulness"] == pytest.approx(0.5)
    # relevancy: judge score 4 / 4 = 1.0
    assert record["generation_metrics"]["answer_relevancy"] == pytest.approx(1.0)
    # claim_precision: 1 correct / 2 generated = 0.5
    assert record["generation_metrics"]["claim_precision"] == pytest.approx(0.5)
    # claim_recall / completeness: 1 covered / 2 reference claims = 0.5
    assert record["generation_metrics"]["claim_recall"] == pytest.approx(0.5)
    assert record["generation_metrics"]["completeness"] == pytest.approx(0.5)
    assert record["generation_metrics"]["claim_f1"] == pytest.approx(0.5)
    # citation_precision: 1 supported / 2 cited = 0.5
    assert record["citation_metrics"]["citation_precision"] == pytest.approx(0.5)
    # citation_recall: (2 claims - 1 unsupported) supported / 2 claims = 0.5
    assert record["citation_metrics"]["citation_recall"] == pytest.approx(0.5)
    assert record["judge_errors"] == []
    assert chat.calls == 5


@pytest.mark.asyncio
async def test_judge_errors_are_collected_without_raising() -> None:
    from scripts.evaluation.judge.schemas import (
        CitationSupportJudgment,
        ClaimCorrectnessJudgment,
        ClaimDecompositionSchema,
        RelevancyJudgment,
    )

    # Faithfulness call fails all three attempts; every other call still runs and
    # succeeds, so one judge_error is recorded but the rest of the record is intact.
    chat = FakeChatModel(
        [
            ClaimDecompositionSchema(claims=["Claim A."]),
            ValueError("bad-1"),
            ValueError("bad-2"),
            ValueError("bad-3"),
            RelevancyJudgment(score=2, reason="partial"),
            ClaimCorrectnessJudgment(
                generated_claim_labels=["correct"],
                covered_reference_indices=[0],
            ),
            CitationSupportJudgment(supported_pair_indices=[]),
        ],
    )
    ask_response = _ask_response(answer="Claim A.", refused=False, citation_count=1)

    record = await _score_generation(chat, _case(), ask_response)

    assert len(record["judge_errors"]) == 1
    assert "bad-3" in record["judge_errors"][0]
    # faithfulness is unscoreable (judge_error), but the other metrics still land.
    assert record["generation_metrics"]["faithfulness"] is None
    assert record["generation_metrics"]["answer_relevancy"] == pytest.approx(0.5)
