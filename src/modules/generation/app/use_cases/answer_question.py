from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.modules.generation.app.context_merge import ContextMerger, RetrievedContextSet
from src.modules.generation.app.dto import AskRequest, AskResult, CitationItem
from src.modules.generation.app.llm_schema import (
    GroundedAnswerSchema,
    MathEvidenceSchema,
)
from src.modules.generation.app.ports import IContextRetriever
from src.modules.generation.app.query_analysis import QueryAnalyzer, QueryPlan
from src.modules.generation.domain.errors import GenerationValidationError
from src.modules.generation.domain.models import ContextChunk
from src.modules.generation.domain.prompt import REFUSAL_ANSWER, GroundedPromptBuilder
from src.shared.app.ports.chat import IChatModel
from src.shared.configs.settings import ChatSettings
from src.shared.observability.metrics import (
    generation_ask_citations,
    generation_ask_contexts,
    generation_ask_duration,
    generation_ask_responses,
    generation_ask_retrieval_queries,
    generation_ask_subquestions,
    generation_context_merge_duration,
    generation_query_analysis_duration,
    generation_retrieval_query_duration,
)
from src.shared.observability.tracing import traced

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)
_FORMULA_MARKERS = (
    "=", "\\frac", "\\sum", "\\prod", "^", "_", "\\sqrt", "\\exp", "\\ln", "\\log", 
    "\\sin", "\\cos", "\\tan", "\\cot", "\\sec", "\\csc", "\\sinh", "\\cosh", "\\tanh", 
    "\\coth", "\\sech", "\\csch", "\\arcsin", "\\arccos", "\\arctan", "\\arccot", "\\arcsec", "\\arccsc", "\\arcsinh", "\\arccosh", "\\arctanh", "\\arccoth", "\\arcsech", "\\arccsch",
    "\\lim", "\\inf", "\\sup", "\\min", "\\max", "\\inf", "\\sup", "\\min", "\\max",
)


@dataclass(frozen=True, slots=True)
class _IntentAnswer:
    intent_index: int
    answer: str = ""
    cited_indices: tuple[int, ...] = ()
    has_math_evidence: bool = False


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
            max_planning_iterations=chat_settings.complex_rag_max_iterations,
        )
        self._context_merger = context_merger or ContextMerger()

    async def execute(self, request: AskRequest) -> AskResult:
        started_at = time.perf_counter()
        outcome = "failure"
        citation_count = 0
        strategy = "single_hop"
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
                strategy = query_plan.strategy
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
                    as_of=request.as_of,
                )
                span.set_attribute("rag.context.count", len(contexts))
                supported_intent_indices = _supported_intent_indices(contexts)
                if supported_intent_indices:
                    span.set_attribute(
                        "rag.query.supported_intent_count",
                        len(supported_intent_indices),
                    )
                if not contexts:
                    outcome = "refused_no_context"
                    return AskResult(
                        query=query,
                        answer=REFUSAL_ANSWER,
                        citations=(),
                        refused=True,
                    )

                with _tracer.start_as_current_span(
                    "generation.answer_synthesis",
                    attributes={
                        "rag.ask.strategy": strategy,
                        "rag.context.count": len(contexts),
                    },
                ):
                    if query_plan.sub_questions:
                        structured = await self._answer_complex_intents(
                            query=query,
                            contexts=contexts,
                            sub_questions=query_plan.sub_questions,
                        )
                    else:
                        system, user = self._prompt.build(
                            query,
                            contexts,
                        )
                        structured = await self._chat.complete_structured(
                            system=system,
                            user=user,
                            schema=GroundedAnswerSchema,
                            temperature=self._chat_settings.temperature,
                        )
                    if query_plan.sub_questions and _missing_supported_intents(
                        structured,
                        contexts,
                        supported_intent_indices,
                    ):
                        outcome = "refused_missing_supported_intents"
                        return AskResult(
                            query=query,
                            answer=REFUSAL_ANSWER,
                            citations=(),
                            refused=True,
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
                    {"outcome": outcome, "strategy": strategy},
                )
                generation_ask_responses.add(
                    1,
                    {"outcome": outcome, "strategy": strategy},
                )
                generation_ask_citations.record(
                    citation_count,
                    {"outcome": outcome, "strategy": strategy},
                )

    async def _build_query_plan(self, query: str) -> QueryPlan:
        started_at = time.perf_counter()
        outcome = "success"
        with _tracer.start_as_current_span("generation.query_analysis") as span:
            try:
                if not self._chat_settings.complex_rag_enabled:
                    outcome = "disabled"
                    plan = QueryPlan.single_hop(reason="complex_rag_disabled")
                else:
                    plan = await self._query_analyzer.analyze(query)
                span.set_attribute("rag.ask.strategy", plan.strategy)
                span.set_attribute(
                    "rag.query.subquestion_count",
                    len(plan.sub_questions),
                )
                generation_ask_subquestions.record(
                    len(plan.sub_questions),
                    {"strategy": plan.strategy},
                )
                return plan
            except Exception as error:
                outcome = "failure"
                span.set_status(Status(StatusCode.ERROR, error.__class__.__name__))
                raise
            finally:
                generation_query_analysis_duration.record(
                    time.perf_counter() - started_at,
                    {"outcome": outcome},
                )

    async def _retrieve_contexts(
        self,
        *,
        query: str,
        query_plan: QueryPlan,
        org_id: int,
        top_k: int,
        document_id: int | None,
        document_version_id: int | None,
        as_of: datetime | None,
    ) -> Sequence[ContextChunk]:
        if not query_plan.is_complex:
            retrieval_query = query_plan.rewritten_query or query
            generation_ask_retrieval_queries.record(
                1,
                {"strategy": query_plan.strategy},
            )
            trace.get_current_span().set_attribute("rag.retrieval.query_count", 1)
            result_set = await self._retrieve_context_set(
                retrieval_query,
                query_kind="original",
                org_id=org_id,
                top_k=top_k,
                document_id=document_id,
                document_version_id=document_version_id,
                as_of=as_of,
            )
            generation_ask_contexts.record(
                len(result_set.contexts),
                {"strategy": query_plan.strategy},
            )
            return result_set.contexts

        final_top_k = (
            top_k
            if top_k != self._chat_settings.top_k
            else self._chat_settings.complex_rag_final_top_k
        )
        per_query_top_k = min(
            self._chat_settings.complex_rag_per_query_top_k,
            final_top_k,
        )
        retrieval_requests = query_plan.retrieval_requests(
            query,
            max_queries=self._chat_settings.complex_rag_max_retrieval_queries,
        )
        generation_ask_retrieval_queries.record(
            len(retrieval_requests),
            {"strategy": query_plan.strategy},
        )
        trace.get_current_span().set_attribute(
            "rag.retrieval.query_count",
            len(retrieval_requests),
        )

        result_sets = await asyncio.gather(
            *(
                self._retrieve_context_set(
                    request.query,
                    query_kind=(
                        "original" if request.intent_index is None else "subquestion"
                    ),
                    intent_index=request.intent_index,
                    org_id=org_id,
                    top_k=per_query_top_k,
                    document_id=document_id,
                    document_version_id=document_version_id,
                    as_of=as_of,
                )
                for request in retrieval_requests
            ),
        )
        started_at = time.perf_counter()
        with _tracer.start_as_current_span(
            "generation.context_merge",
            attributes={
                "rag.ask.strategy": query_plan.strategy,
                "rag.retrieval.query_count": len(retrieval_requests),
            },
        ) as span:
            merge_result = self._context_merger.merge(
                result_sets,
                final_top_k=final_top_k,
            )
            span.set_attribute(
                "rag.context.input_count",
                merge_result.input_context_count,
            )
            span.set_attribute(
                "rag.context.deduplicated_count",
                merge_result.deduplicated_count,
            )
            span.set_attribute("rag.context.count", len(merge_result.contexts))
            generation_context_merge_duration.record(
                time.perf_counter() - started_at,
                {"strategy": query_plan.strategy},
            )
            generation_ask_contexts.record(
                len(merge_result.contexts),
                {"strategy": query_plan.strategy},
            )
            return merge_result.contexts

    async def _retrieve_context_set(
        self,
        query: str,
        *,
        query_kind: str,
        intent_index: int | None = None,
        org_id: int,
        top_k: int,
        document_id: int | None,
        document_version_id: int | None,
        as_of: datetime | None,
    ) -> RetrievedContextSet:
        async with traced(
            _tracer,
            f"generation.retrieve.{query_kind}",
            attributes={
                "rag.retrieval.query_kind": query_kind,
                "rag.retrieval.top_k": top_k,
            },
            duration_metric=generation_retrieval_query_duration,
            metric_attributes={"query_kind": query_kind},
        ) as span:
            contexts = await self._retriever.retrieve(
                query,
                org_id=org_id,
                top_k=top_k,
                document_id=document_id,
                document_version_id=document_version_id,
                as_of=as_of,
            )
            span.set_attribute("rag.context.count", len(contexts))
            return RetrievedContextSet(
                query=query,
                query_kind=query_kind,
                contexts=contexts,
                intent_index=intent_index,
            )

    async def _answer_complex_intents(
        self,
        *,
        query: str,
        contexts: Sequence[ContextChunk],
        sub_questions: Sequence[str],
    ) -> GroundedAnswerSchema:
        intent_results = []
        for intent_index, question in enumerate(sub_questions, start=1):
            intent_results.append(
                await self._answer_one_intent(
                    query=question,
                    intent_index=intent_index,
                    contexts=contexts,
                )
            )
        answered = [result for result in intent_results if result.answer]
        unsupported = [
            result.intent_index for result in intent_results if not result.answer
        ]
        if not answered:
            return GroundedAnswerSchema(
                refused=True,
                answer=REFUSAL_ANSWER,
                cited_indices=[],
                covered_questions=[],
                unsupported_questions=unsupported,
            )

        cited_indices: list[int] = []
        covered_questions: list[int] = []
        answer_parts: list[str] = []
        has_math_evidence = False
        for result in answered:
            answer_parts.append(f"{result.intent_index}. {result.answer}")
            covered_questions.append(result.intent_index)
            has_math_evidence = has_math_evidence or result.has_math_evidence
            for cited_index in result.cited_indices:
                if cited_index not in cited_indices:
                    cited_indices.append(cited_index)
        draft_answer = "\n\n".join(answer_parts)
        final_answer = (
            draft_answer
            if has_math_evidence
            else await self._refine_complex_answer(
                query=query,
                contexts=contexts,
                draft_answer=draft_answer,
                sub_questions=sub_questions,
            )
        )

        return GroundedAnswerSchema(
            refused=False,
            answer=final_answer,
            cited_indices=cited_indices,
            covered_questions=covered_questions,
            unsupported_questions=unsupported,
        )

    async def _answer_one_intent(
        self,
        *,
        query: str,
        intent_index: int,
        contexts: Sequence[ContextChunk],
    ) -> _IntentAnswer:
        indexed_contexts = _contexts_for_intent(contexts, intent_index)
        if not indexed_contexts:
            return _IntentAnswer(intent_index=intent_index)

        local_indexed_contexts = indexed_contexts[:2]
        math_indexed_contexts = _combine_indexed_contexts(
            indexed_contexts[:3],
            _unassigned_contexts(contexts)[:2],
        )
        local_contexts = tuple(context for _, context in local_indexed_contexts)
        math_contexts = tuple(context for _, context in math_indexed_contexts)
        math_evidence = await self._extract_math_evidence(
            query=query,
            contexts=math_contexts,
        )
        async with traced(
            _tracer,
            "generation.answer_intent",
            attributes={
                "rag.query.intent_index": intent_index,
                "rag.context.count": len(local_contexts),
                "rag.math_evidence.present": bool(math_evidence),
            },
        ):
            system, user = self._prompt.build_intent_text_answer(
                query,
                local_contexts,
                math_evidence=math_evidence,
            )
            result = await self._chat.complete(
                system=system,
                user=user,
                temperature=None,
            )
        answer = result.content.strip()
        if answer == REFUSAL_ANSWER:
            return _IntentAnswer(intent_index=intent_index)
        if not answer:
            return _IntentAnswer(intent_index=intent_index)
        cited_indices = local_indexed_contexts
        if math_evidence:
            cited_indices = _combine_indexed_contexts(
                cited_indices, math_indexed_contexts
            )

        return _IntentAnswer(
            intent_index=intent_index,
            answer=answer,
            cited_indices=tuple(index for index, _ in cited_indices),
            has_math_evidence=bool(math_evidence),
        )

    async def _refine_complex_answer(
        self,
        *,
        query: str,
        contexts: Sequence[ContextChunk],
        draft_answer: str,
        sub_questions: Sequence[str],
    ) -> str:
        async with traced(
            _tracer,
            "generation.refine_complex_answer",
            attributes={
                "rag.context.count": len(contexts),
                "rag.query.intent_count": len(sub_questions),
            },
        ):
            system, user = self._prompt.build_final_text_answer(
                query,
                contexts,
                draft_answer=draft_answer,
                sub_questions=sub_questions,
            )
            result = await self._chat.complete(
                system=system,
                user=user,
                temperature=None,
            )
        answer = result.content.strip()
        if not answer or answer == REFUSAL_ANSWER:
            return draft_answer
        return answer

    async def _extract_math_evidence(
        self,
        *,
        query: str,
        contexts: Sequence[ContextChunk],
    ) -> str:
        async with traced(
            _tracer,
            "generation.extract_math_evidence",
            attributes={"rag.context.count": len(contexts)},
        ):
            system, user = self._prompt.build_math_evidence(query, contexts)
            result = await self._chat.complete(
                system=system,
                user=user,
                temperature=None,
            )
        answer = result.content.strip()
        if not answer or answer.upper().rstrip(".") == "NO_MATH_EVIDENCE":
            return ""
        async with traced(
            _tracer,
            "generation.verify_math_evidence",
            attributes={"rag.context.count": len(contexts)},
        ):
            system, user = self._prompt.build_math_evidence_review(
                query,
                contexts,
                candidate=answer,
            )
            verified = await self._chat.complete_structured(
                system=system,
                user=user,
                schema=MathEvidenceSchema,
                temperature=None,
            )
        checked = verified.formula_latex.strip()
        if (
            not verified.has_math_evidence
            or not checked
            or not any(marker in checked for marker in _FORMULA_MARKERS)
        ):
            return ""
        return checked


def _supported_intent_indices(contexts: Sequence[ContextChunk]) -> tuple[int, ...]:
    indices: set[int] = set()
    for context in contexts:
        value = context.metadata.get("answer_intent_indices")
        if value is None:
            value = context.metadata.get("answer_intent_index")
        if value is None:
            continue
        if isinstance(value, int):
            indices.add(value)
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                if isinstance(item, int):
                    indices.add(item)
    return tuple(sorted(index for index in indices if index > 0))


def _contexts_for_intent(
    contexts: Sequence[ContextChunk],
    intent_index: int,
) -> tuple[tuple[int, ContextChunk], ...]:
    return tuple(
        (index, context)
        for index, context in enumerate(contexts, start=1)
        if intent_index in _context_intent_indices(context)
    )


def _unassigned_contexts(
    contexts: Sequence[ContextChunk],
) -> tuple[tuple[int, ContextChunk], ...]:
    return tuple(
        (index, context)
        for index, context in enumerate(contexts, start=1)
        if not _context_intent_indices(context)
    )


def _combine_indexed_contexts(
    primary: Sequence[tuple[int, ContextChunk]],
    fallback: Sequence[tuple[int, ContextChunk]],
) -> tuple[tuple[int, ContextChunk], ...]:
    combined: list[tuple[int, ContextChunk]] = []
    seen: set[int] = set()
    for index, context in (*primary, *fallback):
        if index in seen:
            continue
        seen.add(index)
        combined.append((index, context))
    return tuple(combined)


def _context_intent_indices(context: ContextChunk) -> tuple[int, ...]:
    value = context.metadata.get("answer_intent_indices")
    if value is None:
        value = context.metadata.get("answer_intent_index")
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(item for item in value if isinstance(item, int))
    return ()


def _missing_supported_intents(
    answer: GroundedAnswerSchema,
    contexts: Sequence[ContextChunk],
    supported_intent_indices: Sequence[int],
) -> tuple[int, ...]:
    if answer.refused or not supported_intent_indices:
        return ()
    covered = _covered_intents_from_citations(answer.cited_indices, contexts)
    return tuple(index for index in supported_intent_indices if index not in covered)


def _covered_intents_from_citations(
    cited_indices: Sequence[int],
    contexts: Sequence[ContextChunk],
) -> set[int]:
    covered: set[int] = set()
    for source_number in cited_indices:
        index = source_number - 1
        if index < 0 or index >= len(contexts):
            continue
        value = contexts[index].metadata.get("answer_intent_indices")
        if value is None:
            value = contexts[index].metadata.get("answer_intent_index")
        if isinstance(value, int):
            covered.add(value)
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            covered.update(item for item in value if isinstance(item, int))
    return {index for index in covered if index > 0}
