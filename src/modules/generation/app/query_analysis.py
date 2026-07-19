from __future__ import annotations

import logging
from dataclasses import dataclass

from src.modules.generation.app.llm_schema import QueryAnalysisSchema
from src.shared.app.ports.chat import IChatModel

logger = logging.getLogger(__name__)

_SYSTEM = """
You analyze user questions for a RAG retrieval pipeline.
Return JSON with fields:
- is_complex: boolean
- sub_questions: list of self-contained questions
- reason: short string

Rules:
1. Mark is_complex=true only when the user asks for multiple distinct facts,
   comparisons, steps, causes, or calculations.
2. For simple questions, return is_complex=false and sub_questions=[].
3. For complex questions, produce 2 to 5 self-contained sub-questions.
4. Do not answer the user question.
5. Do not include duplicate sub-questions.
""".strip()


@dataclass(frozen=True, slots=True)
class QueryPlan:
    is_complex: bool
    sub_questions: tuple[str, ...] = ()
    reason: str = ""

    @property
    def strategy(self) -> str:
        return "decomposed" if self.is_complex else "single_hop"

    def retrieval_queries(self, original_query: str, *, max_queries: int) -> tuple[str, ...]:
        queries: list[str] = [original_query]
        seen = {_fingerprint(original_query)}
        for sub_question in self.sub_questions:
            key = _fingerprint(sub_question)
            if not key or key in seen:
                continue
            seen.add(key)
            queries.append(sub_question)
            if len(queries) >= max_queries:
                break
        return tuple(queries)

    @classmethod
    def single_hop(cls, reason: str = "") -> QueryPlan:
        return cls(is_complex=False, sub_questions=(), reason=reason)


class QueryAnalyzer:
    def __init__(
        self,
        chat_model: IChatModel,
        *,
        max_subquestions: int,
        max_tokens: int = 512,
    ) -> None:
        self._chat = chat_model
        self._max_subquestions = max_subquestions
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
        except Exception:
            logger.exception("query analysis failed; falling back to single-hop")
            return QueryPlan.single_hop(reason="analysis_failed")

        sub_questions = _clean_sub_questions(
            result.sub_questions,
            original_query=query,
            max_subquestions=self._max_subquestions,
        )
        is_complex = result.is_complex and len(sub_questions) > 0
        return QueryPlan(
            is_complex=is_complex,
            sub_questions=sub_questions if is_complex else (),
            reason=result.reason.strip(),
        )


def _clean_sub_questions(
    sub_questions: list[str],
    *,
    original_query: str,
    max_subquestions: int,
) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen = {_fingerprint(original_query)}
    for item in sub_questions:
        question = " ".join(item.strip().split())
        key = _fingerprint(question)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(question)
        if len(cleaned) >= max_subquestions:
            break
    return tuple(cleaned)


def _fingerprint(value: str) -> str:
    return value.strip().rstrip("?.!").casefold()
