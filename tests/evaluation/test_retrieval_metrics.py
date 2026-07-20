import pytest
from scripts.evaluation.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

_RELEVANT = {"doc_2", "doc_8", "doc_15"}
_RETRIEVED = ["doc_8", "doc_4", "doc_15", "doc_20", "doc_7"]


def test_recall_at_5_matches_worked_example() -> None:
    assert recall_at_k(_RETRIEVED, _RELEVANT, 5) == pytest.approx(0.6667, abs=1e-4)


def test_precision_at_5_matches_worked_example() -> None:
    assert precision_at_k(_RETRIEVED, _RELEVANT, 5) == pytest.approx(0.4, abs=1e-9)


def test_hit_rate_at_5_matches_worked_example() -> None:
    assert hit_rate_at_k(_RETRIEVED, _RELEVANT, 5) == 1.0


def test_reciprocal_rank_of_first_hit() -> None:
    assert reciprocal_rank(_RETRIEVED, _RELEVANT) == pytest.approx(1.0)


def test_mrr_over_ranks_one_two_five_matches_worked_example() -> None:
    rr_values = [
        reciprocal_rank(["a"], {"a"}),
        reciprocal_rank(["x", "a"], {"a"}),
        reciprocal_rank(["x1", "x2", "x3", "x4", "a"], {"a"}),
    ]
    mrr = sum(rr_values) / len(rr_values)
    assert mrr == pytest.approx(0.5667, abs=1e-4)


def test_empty_relevant_returns_none_not_zero() -> None:
    assert recall_at_k(_RETRIEVED, set(), 5) is None
    assert precision_at_k(_RETRIEVED, set(), 5) is None
    assert hit_rate_at_k(_RETRIEVED, set(), 5) is None
    assert reciprocal_rank(_RETRIEVED, set()) is None
    assert ndcg_at_k(_RETRIEVED, {}, 5) is None


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_precision_divides_by_k_not_by_len_retrieved() -> None:
    short_retrieved = ["doc_8", "doc_15"]
    assert precision_at_k(short_retrieved, _RELEVANT, 5) == pytest.approx(0.4)


def test_duplicate_documents_are_deduplicated_preserving_first_position() -> None:
    retrieved_with_dupes = ["doc_8", "doc_8", "doc_15", "doc_4", "doc_7"]
    assert recall_at_k(retrieved_with_dupes, _RELEVANT, 5) == pytest.approx(
        recall_at_k(_RETRIEVED, _RELEVANT, 5),
    )


def test_ndcg_matches_hand_computed_graded_example() -> None:
    # relevant grades: a=3, b=2, c=1; retrieved order deliberately not ideal.
    grades = {"a": 3, "b": 2, "c": 1}
    retrieved = ["c", "a", "b", "x", "x2"]
    # DCG = 1/log2(2) + 3/log2(3) + 2/log2(4) = 3.892789...
    # IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 4.761860...
    # nDCG = DCG / IDCG = 0.817494...
    assert ndcg_at_k(retrieved, grades, 5) == pytest.approx(0.817494, abs=1e-5)


def test_ndcg_ideal_ranking_uses_full_grades_map_not_only_retrieved_items() -> None:
    grades = {"a": 3, "b": 2, "c": 1}
    perfect_retrieval = ["a", "b", "c"]
    assert ndcg_at_k(perfect_retrieval, grades, 3) == pytest.approx(1.0)
