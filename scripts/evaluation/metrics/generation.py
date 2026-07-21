from __future__ import annotations

from collections.abc import Sequence


def faithfulness(*, claim_count: int, unsupported_count: int) -> float | None:
    """Fraction of claims supported by the retrieved context.

    Undefined (None) for an answer with no claims, not 1.0 - a claimless
    (e.g. refused) answer has nothing to be faithful about.
    """
    if claim_count == 0:
        return None
    return (claim_count - unsupported_count) / claim_count


def claim_precision(generated_claim_labels: Sequence[str]) -> float | None:
    """Fraction of generated claims labelled 'correct' against the reference."""
    if not generated_claim_labels:
        return None
    correct = sum(1 for label in generated_claim_labels if label == "correct")
    return correct / len(generated_claim_labels)


def claim_recall(
    *,
    reference_claim_count: int,
    covered_reference_count: int,
) -> float | None:
    """Fraction of reference claims covered by the generated answer."""
    if reference_claim_count == 0:
        return None
    return covered_reference_count / reference_claim_count


def claim_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def completeness(
    *,
    reference_claim_count: int,
    covered_reference_count: int,
) -> float | None:
    return claim_recall(
        reference_claim_count=reference_claim_count,
        covered_reference_count=covered_reference_count,
    )


def answer_relevancy(score_0_to_4: int | None) -> float | None:
    """Normalizes the judge's 0-4 relevancy rubric score to a 0-1 scale."""
    if score_0_to_4 is None:
        return None
    return score_0_to_4 / 4
