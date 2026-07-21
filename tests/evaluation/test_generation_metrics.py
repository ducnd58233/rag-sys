import pytest
from scripts.evaluation.metrics.generation import (
    answer_relevancy,
    claim_f1,
    claim_precision,
    claim_recall,
    completeness,
    faithfulness,
)


def test_faithfulness_matches_hand_computed_fraction() -> None:
    assert faithfulness(claim_count=4, unsupported_count=1) == pytest.approx(0.75)


def test_faithfulness_is_undefined_for_no_claims() -> None:
    assert faithfulness(claim_count=0, unsupported_count=0) is None


def test_claim_precision_matches_hand_computed_fraction() -> None:
    labels = ["correct", "correct", "incorrect"]
    assert claim_precision(labels) == pytest.approx(0.6667, abs=1e-4)


def test_claim_precision_is_undefined_for_no_generated_claims() -> None:
    assert claim_precision([]) is None


def test_claim_recall_matches_hand_computed_fraction() -> None:
    result = claim_recall(reference_claim_count=5, covered_reference_count=3)
    assert result == pytest.approx(0.6)


def test_claim_recall_is_undefined_for_no_reference_claims() -> None:
    assert claim_recall(reference_claim_count=0, covered_reference_count=0) is None


def test_completeness_equals_claim_recall_by_definition() -> None:
    kwargs = {"reference_claim_count": 5, "covered_reference_count": 3}
    assert completeness(**kwargs) == claim_recall(**kwargs)


def test_claim_f1_matches_hand_computed_harmonic_mean() -> None:
    precision = 2 / 3
    recall = 3 / 5
    # 2 * (2/3) * (3/5) / ((2/3) + (3/5)) = 12/19
    assert claim_f1(precision, recall) == pytest.approx(12 / 19, abs=1e-6)


def test_claim_f1_is_zero_when_both_precision_and_recall_are_zero() -> None:
    assert claim_f1(0.0, 0.0) == 0.0


def test_claim_f1_is_undefined_when_either_input_is_undefined() -> None:
    assert claim_f1(None, 0.5) is None
    assert claim_f1(0.5, None) is None


def test_answer_relevancy_normalizes_zero_to_four_scale() -> None:
    assert answer_relevancy(3) == pytest.approx(0.75)
    assert answer_relevancy(0) == pytest.approx(0.0)
    assert answer_relevancy(4) == pytest.approx(1.0)


def test_answer_relevancy_is_undefined_for_no_score() -> None:
    assert answer_relevancy(None) is None
