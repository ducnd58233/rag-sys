from __future__ import annotations

from src.modules.generation.app.context_merge import ContextMerger, RetrievedContextSet
from src.modules.generation.domain.models import ContextChunk


def _chunk(
    chunk_id: str,
    *,
    score: float,
    document_id: str = "1",
    metadata: dict[str, object] | None = None,
) -> ContextChunk:
    return ContextChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {chunk_id}",
        score=score,
        metadata=metadata or {},
    )


def test_graph_evidenced_chunk_outranks_higher_score_at_equal_coverage() -> None:
    merger = ContextMerger()
    result_sets = (
        RetrievedContextSet(
            query="q",
            query_kind="original",
            contexts=(
                _chunk("plain", score=5.0),
                _chunk("graph", score=1.0, metadata={"graph_evidence": "true"}),
            ),
        ),
    )

    merged = merger.merge(result_sets, final_top_k=2)

    assert [context.chunk_id for context in merged.contexts] == ["graph", "plain"]


def test_multi_query_coverage_still_outranks_single_query_graph_evidence() -> None:
    merger = ContextMerger()
    result_sets = (
        RetrievedContextSet(
            query="q1",
            query_kind="original",
            contexts=(_chunk("consensus", score=1.0),),
        ),
        RetrievedContextSet(
            query="q2",
            query_kind="subquestion",
            intent_index=1,
            contexts=(_chunk("consensus", score=1.0),),
        ),
        RetrievedContextSet(
            query="q3",
            query_kind="subquestion",
            intent_index=1,
            contexts=(_chunk("graph", score=1.0, metadata={"graph_evidence": "true"}),),
        ),
    )

    merged = merger.merge(result_sets, final_top_k=2)

    assert merged.contexts[0].chunk_id == "consensus"


def test_merge_without_graph_metadata_ranks_by_coverage_then_score() -> None:
    merger = ContextMerger()
    result_sets = (
        RetrievedContextSet(
            query="q",
            query_kind="original",
            contexts=(_chunk("low", score=1.0), _chunk("high", score=5.0)),
        ),
    )

    merged = merger.merge(result_sets, final_top_k=2)

    assert [context.chunk_id for context in merged.contexts] == ["high", "low"]
