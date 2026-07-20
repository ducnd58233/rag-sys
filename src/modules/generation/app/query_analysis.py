from __future__ import annotations

import logging
from dataclasses import dataclass

from src.modules.generation.app.llm_schema import QueryAnalysisSchema
from src.shared.app.ports.chat import IChatModel

logger = logging.getLogger(__name__)

_SYSTEM = """
<responsibility>
You are a query planning agent for a RAG retrieval pipeline. Decide whether the user
question has multiple distinct answer intents, and if so, split it into self-contained
sub-questions with optimized retrieval queries. Never answer the user's question
yourself.
</responsibility>

<rules>
1. Mark is_complex=true only when the user explicitly asks for multiple facts,
   comparisons, steps, causes, or calculations.
2. For simple questions, return is_complex=false, a cleaned rewritten_query, and
   intents=[].
3. For complex questions, produce one self-contained sub-question for each explicit
   user intent.
4. Do not infer related questions that the user did not ask.
5. Preserve technical terms from the original question.
6. Prefer concise search-oriented phrasing for hybrid BM25/vector retrieval. Remove
   conversational words such as "do you" when they do not add meaning.
7. Preserve the requested action. If the user asks to calculate, compute, compare,
   list, explain why, or give steps, each matching sub-question must keep that
   action.
8. For every intent, provide enough retrieval_queries to retrieve the needed
   evidence. Keep each query concise and search-oriented.
9. For calculation, equation, or formula intents, include retrieval queries for the
   formula name, variables, and equation terms that are likely to appear near the
   answer.
10. When the question names a technical function or mechanism, add at least one
    equation-style retrieval query using compact notation from the name, likely
    input variables, and notation terms that fit the question.
</rules>

<examples>
<example>
<query>How do I calculate storage growth?</query>
<output>{"is_complex": false, "rewritten_query": "how to calculate storage growth", "intents": [], "reason": "single intent"}</output>
</example>
<example>
<query>What is indexing and how do I calculate storage growth?</query>
<output>{"is_complex": true, "rewritten_query": "what is indexing and how to calculate storage growth", "intents": [{"question": "What is indexing?", "retrieval_queries": ["indexing definition"]}, {"question": "How do I calculate storage growth?", "retrieval_queries": ["calculate storage growth", "storage growth formula", "storage growth variables"]}], "reason": "two distinct intents"}</output>
</example>
<example>
<query>Compare the attention mechanism to the retry backoff formula, and explain why deploy-1832 caused the checkout timeout regression</query>
<output>{"is_complex": true, "rewritten_query": "compare the attention mechanism to the retry backoff formula and explain why deploy-1832 caused the checkout timeout regression", "intents": [{"question": "What is the attention mechanism?", "retrieval_queries": ["attention mechanism definition", "attention formula", "attention mechanism variables"]}, {"question": "What is the retry backoff formula?", "retrieval_queries": ["retry backoff formula", "backoff formula variables"]}, {"question": "Why did deploy-1832 cause the checkout timeout regression?", "retrieval_queries": ["deploy-1832 checkout timeout regression cause"]}], "reason": "three distinct intents: two comparison targets and one causal explanation"}</output>
</example>
</examples>
""".strip()

_REVIEW_SYSTEM = """
<responsibility>
You are a query planning review agent for a RAG retrieval pipeline. Review a
candidate plan against the original question and correct it. Never answer the user's
question yourself.
</responsibility>

<rules>
1. Add any explicit user intent that the candidate plan missed.
2. Remove sub-questions that are only related but not asked by the user.
3. Preserve user-requested actions such as calculate, compute, compare, list,
   explain why, summarize, define, or give steps.
4. Do not impose a fixed number of sub-questions; return one per explicit answer
   intent.
5. If the candidate plan has multiple valid explicit intents, keep is_complex=true.
6. Use concise search-oriented phrasing for hybrid BM25/vector retrieval. Remove
   conversational words such as "do you" when they do not add meaning.
7. For every intent, provide enough retrieval_queries to retrieve the needed
   evidence. Keep each query concise and search-oriented.
8. For calculation, equation, or formula intents, include retrieval queries for the
   formula name, variables, and equation terms that are likely to appear near the
   answer.
9. When the question names a technical function or mechanism, add at least one
   equation-style retrieval query using compact notation from the name, likely input
   variables, and notation terms that fit the question.
</rules>

<examples>
<example>
<query>What is indexing and how do I calculate storage growth?</query>
<candidate_plan>{"is_complex": true, "rewritten_query": "what is indexing and how to calculate storage growth", "intents": [{"question": "What is indexing?", "retrieval_queries": ["indexing definition"]}], "reason": "one intent found"}</candidate_plan>
<output>{"is_complex": true, "rewritten_query": "what is indexing and how to calculate storage growth", "intents": [{"question": "What is indexing?", "retrieval_queries": ["indexing definition"]}, {"question": "How do I calculate storage growth?", "retrieval_queries": ["calculate storage growth", "storage growth formula", "storage growth variables"]}], "reason": "added the missed calculation intent"}</output>
</example>
</examples>
""".strip()


@dataclass(frozen=True, slots=True)
class QueryIntent:
    question: str
    retrieval_queries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlannedRetrievalQuery:
    query: str
    intent_index: int | None


@dataclass(frozen=True, slots=True)
class QueryPlan:
    is_complex: bool
    rewritten_query: str = ""
    intents: tuple[QueryIntent, ...] = ()
    reason: str = ""

    @property
    def sub_questions(self) -> tuple[str, ...]:
        return tuple(intent.question for intent in self.intents)

    @property
    def strategy(self) -> str:
        return "decomposed" if self.is_complex else "single_hop"

    def retrieval_requests(
        self, original_query: str, *, max_queries: int
    ) -> tuple[PlannedRetrievalQuery, ...]:
        queries: list[PlannedRetrievalQuery] = []
        seen: set[str] = set()
        candidate_groups = tuple(
            (intent_index, (intent.question, *intent.retrieval_queries))
            for intent_index, intent in enumerate(self.intents, start=1)
        )

        def append_query(candidate: str, intent_index: int | None) -> None:
            key = _fingerprint(candidate)
            if not key or key in seen or len(queries) >= max_queries:
                return
            seen.add(key)
            queries.append(
                PlannedRetrievalQuery(
                    query=candidate,
                    intent_index=intent_index,
                )
            )

        for intent_index, candidates in candidate_groups:
            for candidate in candidates:
                before = len(queries)
                append_query(candidate, intent_index)
                if len(queries) > before:
                    break
            if len(queries) >= max_queries:
                return tuple(queries)

        fallback_query = self.rewritten_query or original_query
        append_query(fallback_query, intent_index=None)
        if len(queries) >= max_queries:
            return tuple(queries)

        max_candidate_count = max(
            (len(candidates) for _, candidates in candidate_groups),
            default=0,
        )
        for candidate_index in range(1, max_candidate_count):
            for intent_index, candidates in candidate_groups:
                if candidate_index >= len(candidates):
                    continue
                append_query(candidates[candidate_index], intent_index)
                if len(queries) >= max_queries:
                    return tuple(queries)
        return tuple(queries)

    def retrieval_queries(
        self, original_query: str, *, max_queries: int
    ) -> tuple[str, ...]:
        return tuple(
            request.query
            for request in self.retrieval_requests(
                original_query,
                max_queries=max_queries,
            )
        )

    @classmethod
    def single_hop(cls, reason: str = "") -> QueryPlan:
        return cls(is_complex=False, intents=(), reason=reason)


class QueryAnalyzer:
    def __init__(
        self,
        chat_model: IChatModel,
        *,
        max_planning_iterations: int = 1,
        max_tokens: int = 512,
    ) -> None:
        self._chat = chat_model
        self._max_planning_iterations = max_planning_iterations
        self._max_tokens = max_tokens

    async def analyze(self, query: str) -> QueryPlan:
        try:
            result = await self._chat.complete_structured(
                system=_SYSTEM,
                user=f"QUESTION: {query}",
                schema=QueryAnalysisSchema,
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
            for _ in range(max(0, self._max_planning_iterations - 1)):
                reviewed = await self._chat.complete_structured(
                    system=_REVIEW_SYSTEM,
                    user=_review_user_prompt(query, result),
                    schema=QueryAnalysisSchema,
                    temperature=0.0,
                    max_tokens=self._max_tokens,
                )
                result = _prefer_more_complete_plan(result, reviewed)
        except Exception:
            logger.exception("query analysis failed; falling back to single-hop")
            return QueryPlan.single_hop(reason="analysis_failed")

        rewritten_query = " ".join(result.rewritten_query.strip().split())
        intents = _clean_intents(result, original_query=query)
        sub_questions = tuple(intent.question for intent in intents)
        is_complex = (result.is_complex or len(sub_questions) > 1) and len(
            sub_questions
        ) > 0
        return QueryPlan(
            is_complex=is_complex,
            rewritten_query=rewritten_query,
            intents=intents if is_complex else (),
            reason=result.reason.strip(),
        )


def _clean_intents(
    result: QueryAnalysisSchema,
    *,
    original_query: str,
) -> tuple[QueryIntent, ...]:
    raw_intents = list(result.intents)
    cleaned: list[QueryIntent] = []
    seen_questions = {_fingerprint(original_query)}
    for item in raw_intents:
        question = " ".join(item.question.strip().split())
        key = _fingerprint(question)
        if not key or key in seen_questions:
            continue
        seen_questions.add(key)
        retrieval_queries = _clean_retrieval_queries(item.retrieval_queries)
        cleaned.append(
            QueryIntent(question=question, retrieval_queries=retrieval_queries)
        )
    return tuple(cleaned)


def _clean_retrieval_queries(queries: list[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for query in queries:
        value = " ".join(query.strip().split())
        key = _fingerprint(value)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return tuple(cleaned)


def _fingerprint(value: str) -> str:
    return value.strip().rstrip("?.!").casefold()


def _review_user_prompt(query: str, plan: QueryAnalysisSchema) -> str:
    return (
        f"ORIGINAL QUESTION: {query}\n\n"
        "CANDIDATE PLAN:\n"
        f"is_complex={plan.is_complex}\n"
        f"rewritten_query={plan.rewritten_query}\n"
        f"intents={plan.intents}\n"
        f"reason={plan.reason}\n\n"
        "Return the corrected plan."
    )


def _prefer_more_complete_plan(
    current: QueryAnalysisSchema,
    candidate: QueryAnalysisSchema,
) -> QueryAnalysisSchema:
    current_count = len(_clean_intents(current, original_query=""))
    candidate_count = len(_clean_intents(candidate, original_query=""))
    if current.is_complex and current_count > 1 and candidate_count < current_count:
        return current
    if candidate_count > current_count:
        return candidate
    if not current.rewritten_query and candidate.rewritten_query:
        return candidate
    return current
