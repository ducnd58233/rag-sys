from __future__ import annotations

import logging
from collections.abc import Sequence

from src.modules.generation.app.dto import AskRequest, AskResult, CitationItem
from src.modules.generation.app.llm_schema import GroundedAnswerSchema
from src.modules.generation.app.ports import IContextRetriever
from src.modules.generation.domain.errors import GenerationValidationError
from src.modules.generation.domain.models import ContextChunk
from src.modules.generation.domain.prompt import REFUSAL_ANSWER, GroundedPromptBuilder
from src.shared.app.ports.chat import IChatModel
from src.shared.configs.settings import ChatSettings

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._chat_settings = chat_settings
        self._retriever = retriever
        self._chat = chat_model
        self._prompt = prompt_builder or GroundedPromptBuilder()

    async def execute(self, request: AskRequest) -> AskResult:
        query = request.query.strip()
        if not query:
            raise GenerationValidationError("query cannot be empty")
        if request.org_id <= 0:
            raise GenerationValidationError(
                "org_id must be greater than 0",
            )

        top_k = (
            request.top_k if request.top_k is not None else self._chat_settings.top_k
        )
        if top_k < 1:
            raise GenerationValidationError("top_k must be greater than 0")

        contexts = await self._retriever.retrieve(
            query,
            org_id=request.org_id,
            top_k=top_k,
            document_id=request.document_id,
            document_version_id=request.document_version_id,
        )
        if not contexts:
            return AskResult(
                query=query,
                answer=REFUSAL_ANSWER,
                citations=(),
                refused=True,
            )

        system, user = self._prompt.build(query, contexts)
        structured = await self._chat.complete_structured(
            system=system,
            user=user,
            schema=GroundedAnswerSchema,
            temperature=self._chat_settings.temperature,
            max_tokens=self._chat_settings.max_tokens,
        )

        if structured.refused:
            return AskResult(
                query=query,
                answer=REFUSAL_ANSWER,
                citations=(),
                refused=True,
            )

        citations = _citations_from_indices(structured.cited_indices, contexts)
        if not citations:
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
            return AskResult(
                query=query,
                answer=REFUSAL_ANSWER,
                citations=(),
                refused=True,
            )

        return AskResult(
            query=query,
            answer=answer,
            citations=citations,
            refused=False,
        )
