from __future__ import annotations

import logging
import time
import asyncio
from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.modules.generation.app.context_merge import ContextMerger, RetrievedContextSet
from src.modules.generation.app.dto import AskRequest, AskResult, CitationItem
from src.modules.generation.app.llm_schema import GroundedAnswerSchema
from src.modules.generation.app.ports import IContextRetriever
from src.modules.generation.app.query_analysis import QueryAnalyzer, QueryPlan
from src.modules.generation.domain.errors import GenerationValidationError
from src.modules.generation.domain.models import ContextChunk
from src.modules.generation.domain.prompt import REFUSAL_ANSWER, GroundedPromptBuilder
from src.shared.app.ports.chat import IChatModel
from src.shared.configs.settings import ChatSettings
from src.shared.observability.metrics import (
    generation_ask_citations,
    generation_ask_duration,
    generation_ask_responses,
)

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


def _citations_from_indices(
    cited_indices: Sequence[int],
    contexts: Sequence[ContextChunk],
) -> tuple[CitationItem, ...]:
    citations: list[CitationItem] = []
    seen: set[str] = set()

    for source_number in cited_indices:
        index = source_number - 1
        if index < 0 or index >= len(contexts):
            continue
        context = contexts[index]
        if context.chunk_id in seen:
            continue
        seen.add(context.chunk_id)
        citations.append(
            CitationItem(
                chunk_id=context.chunk_id,
                document_id=context.document_id,
                content=context.content,
                score=context.score,
            )
        )
    return tuple(citations)


class AnswerQuestionUseCase:
    def __init__(
        self,
        chat_settings: ChatSettings,
        retriever: IContextRetriever,
        chat_model: IChatModel,
        prompt_builder: GroundedPromptBuilder | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        context_merger: ContextMerger | None = None,
    ) -> None:
        self._chat_settings = chat_settings
        self._retriever = retriever
        self._chat = chat_model
        self._prompt = prompt_builder or GroundedPromptBuilder()
        self._query_analyzer = query_analyzer or QueryAnalyzer(
            chat_model,
            max_subquestions=chat_settings.query_decomposition_max_subquestions,
        )
        self._context_merger = context_merger or ContextMerger()

    async def execute(self, request: AskRequest) -> AskResult:
        started_at = time.perf_counter()
        outcome = "failure"
        citation_count = 0
        query = request.query.strip()
        with _tracer.start_as_current_span("generation.ask") as span:
            try:
                if not query:
                    outcome = "validation_error"
                    raise GenerationValidationError("query cannot be empty")
                if request.org_id <= 0:
                    outcome = "validation_error"
                    raise GenerationValidationError(
                        "org_id must be greater than 0",
                    )

                top_k = (
                    request.top_k
                    if request.top_k is not None
                    else self._chat_settings.top_k
                )
                if top_k < 1:
                    outcome = "validation_error"
                    raise GenerationValidationError("top_k must be greater than 0")

                span.set_attribute("rag.query.length", len(query))
                span.set_attribute("rag.retrieval.top_k", top_k)

                query_plan = await self._build_query_plan(query)
                span.set_attribute("rag.ask.strategy", query_plan.strategy)
                span.set_attribute(
                    "rag.query.subquestion_count",
                    len(query_plan.sub_questions),
                )

                contexts = await self._retrieve_contexts(
                    query=query,
                    query_plan=query_plan,
                    org_id=request.org_id,
                    top_k=top_k,
                    document_id=request.document_id,
                    document_version_id=request.document_version_id,
                )
                span.set_attribute("rag.context.count", len(contexts))
                if not contexts:
                    outcome = "refused_no_context"
                    return AskResult(
                        query=query,
                        answer=REFUSAL_ANSWER,
                        citations=(),
                        refused=True,
                    )

                system, user = self._prompt.build(
                    query,
                    contexts,
                    sub_questions=query_plan.sub_questions,
                )
                structured = await self._chat.complete_structured(
                    system=system,
                    user=user,
                    schema=GroundedAnswerSchema,
                    temperature=self._chat_settings.temperature,
                    max_tokens=self._chat_settings.max_tokens,
                )

                if structured.refused:
                    outcome = "refused_model"
                    return AskResult(
                        query=query,
                        answer=REFUSAL_ANSWER,
                        citations=(),
                        refused=True,
                    )

                citations = _citations_from_indices(structured.cited_indices, contexts)
                citation_count = len(citations)
                span.set_attribute("rag.citation.count", citation_count)
                if not citations:
                    outcome = "refused_no_citations"
                    logger.warning(
                        "answer without valid citations; refusing query_len=%d",
                        len(query),
                    )
                    return AskResult(
                        query=query,
                        answer=REFUSAL_ANSWER,
                        citations=(),
                        refused=True,
                    )

                answer = structured.answer.strip() or REFUSAL_ANSWER
                if answer == REFUSAL_ANSWER:
                    outcome = "refused_empty_answer"
                    return AskResult(
                        query=query,
                        answer=REFUSAL_ANSWER,
                        citations=(),
                        refused=True,
                    )

                outcome = "answered"
                return AskResult(
                    query=query,
                    answer=answer,
                    citations=citations,
                    refused=False,
                )
            except Exception as error:
                span.set_status(Status(StatusCode.ERROR, error.__class__.__name__))
                raise
            finally:
                span.set_attribute("rag.ask.outcome", outcome)
                generation_ask_duration.record(
                    time.perf_counter() - started_at,
                    {"outcome": outcome},
                )
                generation_ask_responses.add(1, {"outcome": outcome})
                generation_ask_citations.record(
                    citation_count,
                    {"outcome": outcome},
                )

    async def _build_query_plan(self, query: str) -> QueryPlan:
        if not self._chat_settings.complex_rag_enabled:
            return QueryPlan.single_hop(reason="complex_rag_disabled")
        return await self._query_analyzer.analyze(query)

    async def _retrieve_contexts(
        self,
        *,
        query: str,
        query_plan: QueryPlan,
        org_id: int,
        top_k: int,
        document_id: int | None,
        document_version_id: int | None,
    ) -> Sequence[ContextChunk]:
        if not query_plan.is_complex:
            return await self._retriever.retrieve(
                query,
                org_id=org_id,
                top_k=top_k,
                document_id=document_id,
                document_version_id=document_version_id,
            )

        final_top_k = (
            top_k
            if top_k != self._chat_settings.top_k
            else self._chat_settings.complex_rag_final_top_k
        )
        per_query_top_k = min(
            self._chat_settings.complex_rag_per_query_top_k,
            final_top_k,
        )
        retrieval_queries = query_plan.retrieval_queries(
            query,
            max_queries=self._chat_settings.complex_rag_max_retrieval_queries,
        )

        result_sets = await asyncio.gather(
            *(
                self._retrieve_context_set(
                    retrieval_query,
                    query_kind="original" if index == 0 else "subquestion",
                    org_id=org_id,
                    top_k=per_query_top_k,
                    document_id=document_id,
                    document_version_id=document_version_id,
                )
                for index, retrieval_query in enumerate(retrieval_queries)
            ),
        )
        return self._context_merger.merge(
            result_sets,
            final_top_k=final_top_k,
        ).contexts

    async def _retrieve_context_set(
        self,
        query: str,
        *,
        query_kind: str,
        org_id: int,
        top_k: int,
        document_id: int | None,
        document_version_id: int | None,
    ) -> RetrievedContextSet:
        contexts = await self._retriever.retrieve(
            query,
            org_id=org_id,
            top_k=top_k,
            document_id=document_id,
            document_version_id=document_version_id,
        )
        return RetrievedContextSet(
            query=query,
            query_kind=query_kind,
            contexts=contexts,
        )
