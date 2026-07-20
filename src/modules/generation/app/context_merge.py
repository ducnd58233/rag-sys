from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.modules.generation.domain.models import ContextChunk


@dataclass(frozen=True, slots=True)
class RetrievedContextSet:
    query: str
    query_kind: str
    contexts: Sequence[ContextChunk]
    intent_index: int | None = None


@dataclass(frozen=True, slots=True)
class ContextMergeResult:
    contexts: tuple[ContextChunk, ...]
    input_context_count: int
    deduplicated_count: int
    supported_intent_indices: tuple[int, ...] = ()


@dataclass(slots=True)
class _Candidate:
    context: ContextChunk
    first_seen: int
    query_keys: set[str] = field(default_factory=set)
    intent_indices: set[int] = field(default_factory=set)

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
                candidate_key = _candidate_key(context)
                candidate = candidates.get(candidate_key)
                if candidate is None:
                    candidates[candidate_key] = _Candidate(
                        context=_with_retrieval_metadata(context, result_set),
                        first_seen=seen_order,
                        query_keys={query_key},
                        intent_indices=_intent_indices(result_set),
                    )
                    seen_order += 1
                    continue

                candidate.query_keys.add(query_key)
                candidate.intent_indices.update(_intent_indices(result_set))
                if context.score > candidate.context.score:
                    candidate.context = _with_retrieval_metadata(context, result_set)

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                item.coverage_count,
                _is_graph_evidence(item.context),
                item.context.score,
                -item.first_seen,
            ),
            reverse=True,
        )
        selected = ordered[:final_top_k]
        contexts = tuple(
            _with_candidate_metadata(item.context, item) for item in selected
        )
        supported_intent_indices = tuple(
            sorted(
                {
                    index
                    for item in selected
                    for index in item.intent_indices
                    if index > 0
                }
            )
        )
        return ContextMergeResult(
            contexts=contexts,
            input_context_count=input_context_count,
            deduplicated_count=len(candidates),
            supported_intent_indices=supported_intent_indices,
        )


def _is_graph_evidence(context: ContextChunk) -> bool:
    return str(context.metadata.get("graph_evidence", "")).lower() == "true"


def _query_key(query: str) -> str:
    return " ".join(query.strip().casefold().split())


def _candidate_key(context: ContextChunk) -> str:
    content_key = _content_key(context.content)
    if content_key:
        return f"content:{content_key}"
    return f"chunk:{context.chunk_id}"


def _content_key(content: str) -> str:
    return re.sub(r"\W+", " ", content.casefold()).strip()


def _intent_indices(result_set: RetrievedContextSet) -> set[int]:
    index = result_set.intent_index
    return {index} if index is not None and index > 0 else set()


def _with_retrieval_metadata(
    context: ContextChunk,
    result_set: RetrievedContextSet,
) -> ContextChunk:
    metadata = dict(context.metadata)
    metadata["retrieval_query_kind"] = result_set.query_kind
    metadata["retrieval_query"] = result_set.query
    if result_set.intent_index is not None:
        metadata["answer_intent_index"] = result_set.intent_index
    return ContextChunk(
        chunk_id=context.chunk_id,
        document_id=context.document_id,
        content=context.content,
        score=context.score,
        metadata=metadata,
    )


def _with_candidate_metadata(
    context: ContextChunk, candidate: _Candidate
) -> ContextChunk:
    metadata = dict(context.metadata)
    metadata["retrieval_query_count"] = candidate.coverage_count
    if candidate.intent_indices:
        metadata["answer_intent_indices"] = tuple(sorted(candidate.intent_indices))
    return ContextChunk(
        chunk_id=context.chunk_id,
        document_id=context.document_id,
        content=context.content,
        score=context.score,
        metadata=metadata,
    )
