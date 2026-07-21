from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from src.shared.app.retry import RetryPolicy
from src.shared.configs.settings import ChatSettings, OllamaSettings
from src.shared.infra.chat.factory import ChatModelFactory
from src.shared.infra.chat.langchain import LangChainChatModel


class CapturingChatClient:
    def __init__(self) -> None:
        self.bind_kwargs: dict[str, object] | None = None

    def bind(self, **kwargs: object) -> CapturingChatClient:
        self.bind_kwargs = kwargs
        return self

    async def ainvoke(self, messages: object) -> AIMessage:
        del messages
        return AIMessage(content="ok")


@pytest.mark.asyncio
async def test_ollama_chat_binds_num_predict_instead_of_max_tokens() -> None:
    client = CapturingChatClient()
    model = LangChainChatModel(
        client,  # type: ignore[arg-type]
        model_name="qwen2.5",
        provider_name="ollama",
        retry_policy=RetryPolicy(max_attempts=1),
    )

    await model.complete(system="s", user="u", max_tokens=8)

    assert client.bind_kwargs == {"num_predict": 8}


@pytest.mark.asyncio
async def test_vllm_chat_binds_max_tokens() -> None:
    client = CapturingChatClient()
    model = LangChainChatModel(
        client,  # type: ignore[arg-type]
        model_name="qwen2.5",
        provider_name="vllm",
        retry_policy=RetryPolicy(max_attempts=1),
    )

    await model.complete(system="s", user="u", max_tokens=8)

    assert client.bind_kwargs == {"max_tokens": 8}


def test_ollama_chat_factory_sets_num_predict() -> None:
    settings = ChatSettings(
        provider="ollama",
        model="qwen2.5:1.5b",
        max_tokens=64,
        ollama=OllamaSettings(url="http://localhost:11434"),
    )

    model = ChatModelFactory.from_settings(
        settings,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    client = model._client  # noqa: SLF001

    assert client.num_predict == 64
