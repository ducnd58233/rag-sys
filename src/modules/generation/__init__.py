from __future__ import annotations

from dataclasses import dataclass

from src.modules.generation.app.context_merge import ContextMerger
from src.modules.generation.app.query_analysis import QueryAnalyzer, QueryPlan
from src.modules.generation.app.use_cases.answer_question import AnswerQuestionUseCase
from src.modules.generation.domain.prompt import GroundedPromptBuilder
from src.modules.generation.infra.retrieval_adapter import RetrieveUseCaseAdapter
from src.modules.retrieval.app.use_cases.retrieve import RetrieveUseCase
from src.shared.configs.settings import Settings
from src.shared.infra.chat import ChatModelFactory

__all__ = [
    "AnswerQuestionUseCase",
    "GenerationComponentFactory",
    "GenerationComponents",
    "ContextMerger",
    "QueryAnalyzer",
    "QueryPlan",
]


@dataclass(frozen=True, slots=True)
class GenerationComponents:
    answer_question: AnswerQuestionUseCase


class GenerationComponentFactory:
    @staticmethod
    def build(
        *,
        settings: Settings,
        retrieve_use_case: RetrieveUseCase,
    ) -> GenerationComponents:
        chat = settings.chat
        chat_model = ChatModelFactory.from_settings(chat)
        return GenerationComponents(
            answer_question=AnswerQuestionUseCase(
                chat_settings=chat,
                retriever=RetrieveUseCaseAdapter(retrieve_use_case),
                chat_model=chat_model,
                prompt_builder=GroundedPromptBuilder(),
                query_analyzer=QueryAnalyzer(
                    chat_model,
                    max_planning_iterations=chat.complex_rag_max_iterations,
                ),
                context_merger=ContextMerger(),
            ),
        )
