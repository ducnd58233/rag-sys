from __future__ import annotations

import pytest

from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.infra.fusion.reciprocal_rank import ReciprocalRankFusion


def test_unweighted_fusion_keeps_existing_scores() -> None:
    fusion = ReciprocalRankFusion(rank_constant=60)
    ranked_lists = [
        (_hit("a"), _hit("b")),
        (_hit("b"), _hit("c")),
    ]

    result = fusion.fuse(ranked_lists, top_k=3)

    assert [(hit.chunk_id, hit.score) for hit in result] == [
        ("b", (1 / 62) + (1 / 61)),
        ("a", 1 / 61),
        ("c", 1 / 62),
    ]


def test_explicit_equal_weights_match_unweighted_result() -> None:
    fusion = ReciprocalRankFusion(rank_constant=60)
    ranked_lists = [
        (_hit("a"), _hit("b")),
        (_hit("b"), _hit("c")),
    ]

    assert fusion.fuse(ranked_lists, top_k=3) == fusion.fuse(
        ranked_lists,
        top_k=3,
        weights=(1.0, 1.0),
    )


def test_weighted_fusion_sums_duplicate_contributions() -> None:
    fusion = ReciprocalRankFusion(rank_constant=60)
    ranked_lists = [
        (_hit("a"), _hit("b")),
        (_hit("b"), _hit("c")),
    ]

    result = fusion.fuse(ranked_lists, top_k=3, weights=(0.2, 1.0))

    assert [hit.chunk_id for hit in result] == ["b", "c", "a"]
    assert len({hit.chunk_id for hit in result}) == len(result)


def test_fusion_rejects_invalid_weights() -> None:
    fusion = ReciprocalRankFusion()

    with pytest.raises(ValueError, match="weights must match"):
        fusion.fuse([(_hit("a"),)], top_k=1, weights=(1.0, 1.0))

    with pytest.raises(ValueError, match="negative"):
        fusion.fuse([(_hit("a"),)], top_k=1, weights=(-1.0,))


def _hit(chunk_id: str) -> HitChunk:
    return HitChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"content {chunk_id}",
        score=1.0,
        metadata={},
    )
