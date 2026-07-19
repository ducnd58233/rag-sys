from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from src.modules.generation.domain.models import ContextChunk


@dataclass(frozen=True, slots=True)
class RetrievedContextSet:
    query: str
    query_kind: str
    contexts: Sequence[ContextChunk]


@dataclass(frozen=True, slots=True)
class ContextMergeResult:
    contexts: tuple[ContextChunk, ...]
    input_context_count: int
    deduplicated_count: int


@dataclass(slots=True)
class _Candidate:
    context: ContextChunk
    first_seen: int
    query_keys: set[str] = field(default_factory=set)

    @property
    def coverage_count(self) -> int:
        return len(self.query_keys)


class ContextMerger:
    def merge(
        self,
        result_sets: Sequence[RetrievedContextSet],
        *,
        final_top_k: int,
    ) -> ContextMergeResult:
        candidates: dict[str, _Candidate] = {}
        input_context_count = 0
        seen_order = 0

        for result_set in result_sets:
            query_key = _query_key(result_set.query)
            for context in result_set.contexts:
                input_context_count += 1
                candidate = candidates.get(context.chunk_id)
                if candidate is None:
                    candidates[context.chunk_id] = _Candidate(
                        context=context,
                        first_seen=seen_order,
                        query_keys={query_key},
                    )
                    seen_order += 1
                    continue

                candidate.query_keys.add(query_key)
                if context.score > candidate.context.score:
                    candidate.context = context

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                item.coverage_count,
                item.context.score,
                -item.first_seen,
            ),
            reverse=True,
        )
        contexts = tuple(item.context for item in ordered[:final_top_k])
        return ContextMergeResult(
            contexts=contexts,
            input_context_count=input_context_count,
            deduplicated_count=len(candidates),
        )


def _query_key(query: str) -> str:
    return " ".join(query.strip().casefold().split())
