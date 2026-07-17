from __future__ import annotations

from src.modules.generation.app.use_cases.answer_question import AnswerQuestionUseCase
from src.modules.generation.domain.prompt import GroundedPromptBuilder
from src.modules.generation.infra.retrieval_adapter import RetrieveUseCaseAdapter
from src.modules.retrieval.app.use_cases.retrieve import RetrieveUseCase
from src.shared.configs.settings import Settings
from src.shared.infra.chat import ChatModelFactory

__all__ = ["GenerationComponentFactory", "AnswerQuestionUseCase"]

class GenerationComponentFactory:
    @staticmethod
    def build_answer_question_use_case(
        settings: Settings,
        retrieve_use_case: RetrieveUseCase,
    ) -> AnswerQuestionUseCase:
        chat = settings.chat
        return AnswerQuestionUseCase(
            chat_settings=chat,
            retriever=RetrieveUseCaseAdapter(retrieve_use_case),
            chat_model=ChatModelFactory.from_settings(chat),
            prompt_builder=GroundedPromptBuilder(),
        )