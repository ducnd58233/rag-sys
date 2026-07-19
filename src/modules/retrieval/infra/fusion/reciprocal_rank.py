from __future__ import annotations

from collections.abc import Sequence

from src.modules.retrieval.domain.models import HitChunk


class ReciprocalRankFusion:
    def __init__(self, rank_constant: int = 60) -> None:
        self._rank_constant = rank_constant

    def fuse(
        self,
        ranked_lists: Sequence[Sequence[HitChunk]],
        *,
        top_k: int,
    ) -> tuple[HitChunk, ...]:
        if not ranked_lists:
            raise ValueError("ranked_lists cannot be empty")

        if self._rank_constant < 1:
            raise ValueError("rank_constant must be greater than 0")

        if top_k < 1:
            raise ValueError("top_k must be greater than 0")

        fused: dict[str, float] = {}
        by_id: dict[str, HitChunk] = {}

        for ranked in ranked_lists:
            for rank, chunk in enumerate(ranked, start=1):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + (
                    1.0 / (self._rank_constant + rank)
                )
                by_id[chunk.chunk_id] = chunk

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        results: list[HitChunk] = []
        for chunk_id, score in ordered[:top_k]:
            src = by_id[chunk_id]
            results.append(
                HitChunk(
                    chunk_id=src.chunk_id,
                    document_id=src.document_id,
                    content=src.content,
                    score=score,
                    metadata=src.metadata,
                )
            )
        return tuple(results)
