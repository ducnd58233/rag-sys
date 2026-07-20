"""Load and validate the golden evaluation dataset (docs/rag-evaluation/SPEC.md FR-EVAL-1).

A malformed case must fail loudly here, at load time, not silently produce a wrong metric
later - a metric computed over a case with an inconsistent schema looks exactly like a
metric computed over a valid one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Split = Literal["dev", "test"]


class DatasetValidationError(ValueError):
    def __init__(self, case_id: str, reason: str) -> None:
        super().__init__(f"case {case_id!r}: {reason}")
        self.case_id = case_id
        self.reason = reason


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    question: str
    reference_answer: str | None
    reference_claims: tuple[str, ...]
    relevant_document_ids: tuple[str, ...]
    relevance_grades: dict[str, int]
    answerable: bool
    difficulty: str
    split: Split
    tags: tuple[str, ...]
    reviewed_by: str | None
    expected_behavior: str | None = None


def _validate(raw: dict[str, object]) -> None:
    case_id = str(raw.get("id", "<missing id>"))

    if raw.get("split") not in ("dev", "test"):
        raise DatasetValidationError(case_id, "split must be 'dev' or 'test'")

    answerable = raw.get("answerable")
    if not isinstance(answerable, bool):
        raise DatasetValidationError(case_id, "answerable must be a bool")

    relevant_document_ids = raw.get("relevant_document_ids")
    if not isinstance(relevant_document_ids, list):
        raise DatasetValidationError(case_id, "relevant_document_ids must be a list")

    reference_claims = raw.get("reference_claims")
    if not isinstance(reference_claims, list):
        raise DatasetValidationError(case_id, "reference_claims must be a list")

    relevance_grades = raw.get("relevance_grades", {})
    if not isinstance(relevance_grades, dict):
        raise DatasetValidationError(case_id, "relevance_grades must be an object")
    unknown_grade_keys = set(relevance_grades) - set(relevant_document_ids)
    if unknown_grade_keys:
        raise DatasetValidationError(
            case_id,
            f"relevance_grades keys {sorted(unknown_grade_keys)} are not in "
            "relevant_document_ids",
        )

    if answerable:
        if not reference_claims:
            raise DatasetValidationError(
                case_id,
                "answerable=true requires a non-empty reference_claims",
            )
        if not relevant_document_ids:
            raise DatasetValidationError(
                case_id,
                "answerable=true requires a non-empty relevant_document_ids",
            )
    else:
        if raw.get("reference_answer") is not None:
            raise DatasetValidationError(
                case_id,
                "answerable=false requires reference_answer to be null",
            )
        if reference_claims:
            raise DatasetValidationError(
                case_id,
                "answerable=false requires an empty reference_claims",
            )
        if relevant_document_ids:
            raise DatasetValidationError(
                case_id,
                "answerable=false requires an empty relevant_document_ids",
            )
        if raw.get("expected_behavior") != "abstain":
            raise DatasetValidationError(
                case_id,
                "answerable=false requires expected_behavior='abstain'",
            )


def _to_case(raw: dict[str, object]) -> EvalCase:
    _validate(raw)
    return EvalCase(
        id=str(raw["id"]),
        question=str(raw["question"]),
        reference_answer=raw.get("reference_answer"),  # type: ignore[arg-type]
        reference_claims=tuple(raw.get("reference_claims", [])),  # type: ignore[arg-type]
        relevant_document_ids=tuple(
            raw.get("relevant_document_ids", []),  # type: ignore[arg-type]
        ),
        relevance_grades=dict(raw.get("relevance_grades", {})),  # type: ignore[arg-type]
        answerable=bool(raw["answerable"]),
        difficulty=str(raw.get("difficulty", "")),
        split=raw["split"],  # type: ignore[assignment]
        tags=tuple(raw.get("tags", [])),  # type: ignore[arg-type]
        reviewed_by=raw.get("reviewed_by"),  # type: ignore[arg-type]
        expected_behavior=raw.get("expected_behavior"),  # type: ignore[arg-type]
    )


def load_dataset(path: Path, *, split: Split | None = None) -> tuple[EvalCase, ...]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetValidationError(
                    f"<line {line_no}>",
                    f"invalid JSON: {error}",
                ) from error
            cases.append(_to_case(raw))

    if split is not None:
        cases = [case for case in cases if case.split == split]
    return tuple(cases)


def distribution(cases: Sequence[EvalCase]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.difficulty] = counts.get(case.difficulty, 0) + 1
    return counts
