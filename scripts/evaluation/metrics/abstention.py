from __future__ import annotations

from collections.abc import Sequence


def confusion_matrix(records: Sequence[tuple[bool, bool]]) -> dict[str, int]:
    """Builds the FR-EVAL-5 confusion matrix from (answerable, refused) pairs."""
    true_positive = sum(
        1 for answerable, refused in records if answerable and not refused
    )
    false_negative = sum(1 for answerable, refused in records if answerable and refused)
    true_negative = sum(
        1 for answerable, refused in records if not answerable and refused
    )
    false_positive = sum(
        1 for answerable, refused in records if not answerable and not refused
    )
    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
    }


def abstention_precision(matrix: dict[str, int]) -> float | None:
    """Of every case the model abstained on, the fraction it was right to."""
    predicted_abstain = matrix["true_negative"] + matrix["false_negative"]
    if predicted_abstain == 0:
        return None
    return matrix["true_negative"] / predicted_abstain


def abstention_recall(matrix: dict[str, int]) -> float | None:
    """Of every truly unanswerable case, the fraction the model abstained on."""
    actual_unanswerable = matrix["true_negative"] + matrix["false_positive"]
    if actual_unanswerable == 0:
        return None
    return matrix["true_negative"] / actual_unanswerable


def abstention_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def unsupported_answer_rate(matrix: dict[str, int]) -> float | None:
    """Of every case the model chose to answer, the fraction that were hallucinated."""
    answered = matrix["true_positive"] + matrix["false_positive"]
    if answered == 0:
        return None
    return matrix["false_positive"] / answered
