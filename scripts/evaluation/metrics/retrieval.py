from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set


def _dedupe_preserve_order(retrieved: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for doc_id in retrieved:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(doc_id)
    return deduped


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Set[str],
    k: int,
) -> float | None:
    """Fraction of `relevant` present in the top k of `retrieved`.

    Returns None when `relevant` is empty (an unanswerable case) - the value is
    undefined, not zero, and averaging it as 0.0 would poison the mean with cases
    that were never supposed to have a relevant document.
    """
    if not relevant:
        return None
    top_k = _dedupe_preserve_order(retrieved)[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(
    retrieved: Sequence[str],
    relevant: Set[str],
    k: int,
) -> float | None:
    """Fraction of the top k slots that were relevant.

    Divides by k, not by len(retrieved): retrieving fewer than k items is itself a
    precision cost, not a reason to inflate the score.
    """
    if not relevant:
        return None
    top_k = _dedupe_preserve_order(retrieved)[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def hit_rate_at_k(
    retrieved: Sequence[str],
    relevant: Set[str],
    k: int,
) -> float | None:
    if not relevant:
        return None
    top_k = _dedupe_preserve_order(retrieved)[:k]
    return 1.0 if any(doc_id in relevant for doc_id in top_k) else 0.0


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: Set[str],
) -> float | None:
    if not relevant:
        return None
    for rank, doc_id in enumerate(_dedupe_preserve_order(retrieved), start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    grades: Mapping[str, int],
    k: int,
) -> float | None:
    """Normalized discounted cumulative gain over graded relevance (0-3).

    The ideal ranking is computed from the full `grades` map, not only from the
    documents that happened to be retrieved - otherwise a retriever that finds
    nothing would trivially score a perfect nDCG of 1.0.
    """
    if not grades:
        return None
    top_k = _dedupe_preserve_order(retrieved)[:k]
    dcg = sum(
        grades.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
    )
    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(
        grade / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1)
    )
    if idcg == 0:
        return 0.0
    return dcg / idcg
