from __future__ import annotations

import os
from dataclasses import dataclass

from src.shared.app.ports import IChatModel
from src.shared.configs.settings import ChatSettings
from src.shared.infra.chat.factory import ChatModelFactory

_JUDGE_PROVIDER_ENV = "EVAL_JUDGE_PROVIDER"
_JUDGE_MODEL_ENV = "EVAL_JUDGE_MODEL"
JUDGE_TEMPERATURE = 0.0


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    provider: str
    model: str
    temperature: float
    self_preference_risk: bool


def resolve_judge_config(app_chat_settings: ChatSettings | None = None) -> JudgeConfig:
    app_settings = app_chat_settings or ChatSettings()
    provider = os.environ.get(_JUDGE_PROVIDER_ENV, app_settings.provider)
    model = os.environ.get(_JUDGE_MODEL_ENV, app_settings.model)
    return JudgeConfig(
        provider=provider,
        model=model,
        temperature=JUDGE_TEMPERATURE,
        self_preference_risk=(
            provider == app_settings.provider and model == app_settings.model
        ),
    )


def build_judge_chat_model(config: JudgeConfig) -> IChatModel:
    settings = ChatSettings(
        provider=config.provider,
        model=config.model,
        temperature=config.temperature,
    )
    return ChatModelFactory.from_settings(settings)
