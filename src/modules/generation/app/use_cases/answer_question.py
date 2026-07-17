from __future__ import annotations

import logging

from src.modules.generation.app.dto import AskRequest, AskResult, CitationItem
from src.modules.generation.app.ports import IContextRetriever
from src.modules.generation.domain.errors import GenerationValidationError
from src.modules.generation.domain.prompt import GroundedPromptBuilder
from src.shared.app.ports.chat import IChatModel
from src.shared.configs.settings import ChatSettings

logger = logging.getLogger(__name__)
_REFUSAL = "I could not find relevant information to answer this question."


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

        top_k = request.top_k if request.top_k is not None else self._chat_settings.top_k
        if top_k < 1:
            raise GenerationValidationError("top_k must be greater than 0")

        contexts = await self._retriever.retrieve(
            query, top_k=top_k, document_id=request.document_id
        )
        if not contexts:
            return AskResult(
                query=query, 
                answer=_REFUSAL, 
                citations=(),
                refused=True
            )

        system, user = self._prompt.build(query, contexts)
        result = await self._chat.complete(system=system, user=user)

        return AskResult(
            query=query,
            answer=result.content,
            citations=tuple(
                CitationItem(chunk_id=c.chunk_id, document_id=c.document_id)
                for c in contexts
            ),
            refused=False,
        )