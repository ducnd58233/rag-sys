import pytest
from scripts.evaluation.metrics.citation import citation_precision, citation_recall

# SPEC.md FR-EVAL-4 worked example: 5 claims, 4 cited, 3 supported -> recall
# 0.80, precision 0.75.


def test_citation_precision_matches_spec_worked_example() -> None:
    result = citation_precision(cited_count=4, supported_count=3)
    assert result == pytest.approx(0.75)


def test_citation_recall_matches_spec_worked_example() -> None:
    result = citation_recall(
        claims_requiring_evidence_count=5,
        claims_with_valid_citation_count=4,
    )
    assert result == pytest.approx(0.80)


def test_citation_precision_is_undefined_for_no_citations() -> None:
    assert citation_precision(cited_count=0, supported_count=0) is None


def test_citation_recall_is_undefined_for_no_evidence_requiring_claims() -> None:
    result = citation_recall(
        claims_requiring_evidence_count=0,
        claims_with_valid_citation_count=0,
    )
    assert result is None
