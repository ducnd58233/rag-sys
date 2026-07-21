import pytest
from scripts.evaluation.metrics.abstention import (
    abstention_f1,
    abstention_precision,
    abstention_recall,
    confusion_matrix,
    unsupported_answer_rate,
)

# (answerable, refused): 2 true positives, 1 false negative, 3 true negatives,
# 1 false positive - all four quadrants represented.
_RECORDS = [
    (True, False),
    (True, False),
    (True, True),
    (False, True),
    (False, True),
    (False, True),
    (False, False),
]


def test_confusion_matrix_counts_all_four_quadrants() -> None:
    matrix = confusion_matrix(_RECORDS)

    assert matrix == {
        "true_positive": 2,
        "false_negative": 1,
        "true_negative": 3,
        "false_positive": 1,
    }


def test_abstention_precision_matches_hand_computed_value() -> None:
    matrix = confusion_matrix(_RECORDS)
    # true_negative / (true_negative + false_negative) = 3 / 4
    assert abstention_precision(matrix) == pytest.approx(0.75)


def test_abstention_recall_matches_hand_computed_value() -> None:
    matrix = confusion_matrix(_RECORDS)
    # true_negative / (true_negative + false_positive) = 3 / 4
    assert abstention_recall(matrix) == pytest.approx(0.75)


def test_abstention_f1_matches_hand_computed_value() -> None:
    matrix = confusion_matrix(_RECORDS)
    assert abstention_f1(
        abstention_precision(matrix),
        abstention_recall(matrix),
    ) == pytest.approx(0.75)


def test_unsupported_answer_rate_matches_hand_computed_value() -> None:
    matrix = confusion_matrix(_RECORDS)
    # false_positive / (true_positive + false_positive) = 1 / 3
    assert unsupported_answer_rate(matrix) == pytest.approx(1 / 3, abs=1e-6)


def test_abstention_precision_is_undefined_when_nothing_was_abstained_on() -> None:
    matrix = confusion_matrix([(True, False), (False, False)])
    assert abstention_precision(matrix) is None


def test_abstention_recall_is_undefined_when_nothing_was_truly_unanswerable() -> None:
    matrix = confusion_matrix([(True, False), (True, True)])
    assert abstention_recall(matrix) is None


def test_unsupported_answer_rate_is_undefined_when_nothing_was_answered() -> None:
    matrix = confusion_matrix([(True, True), (False, True)])
    assert unsupported_answer_rate(matrix) is None


def test_safety_refusals_must_be_pre_filtered_by_the_caller_before_scoring() -> None:
    """PLAN.md's abstention classification: a safety refusal is excluded from
    abstention metrics entirely, not counted as an abstention. confusion_matrix()
    has no refusal-reason signal (the API's `refused` is a single boolean, and
    `safety_decision` does not exist yet - see datasets/README.md and
    docs/rag-evaluation/PLAN.md section D), so the caller must drop any
    safety-refused case before building records, as this test's absence of such
    a case demonstrates: every record here is either a real answer or a
    evidence-abstention, never a safety refusal.
    """
    matrix = confusion_matrix(_RECORDS)
    assert sum(matrix.values()) == len(_RECORDS)
