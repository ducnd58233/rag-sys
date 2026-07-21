from __future__ import annotations

import httpx
import pytest
from langchain_core.messages import AIMessage

from src.shared.app.retry import RetryPolicy
from src.shared.infra.chat.langchain import LangChainChatModel
from src.shared.infra.embedding.langchain import LangChainEmbeddingModel


class FlakyEmbedClient:
    def __init__(self, failures_before_success: int) -> None:
        self._remaining = failures_before_success
        self.calls = 0

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise httpx.ReadError("peer closed")
        return [[0.1, 0.2] for _ in texts]


class FlakyChatClient:
    def __init__(self, failures_before_success: int) -> None:
        self._remaining = failures_before_success
        self.calls = 0

    def bind(self, **kwargs: object) -> FlakyChatClient:
        del kwargs
        return self

    def with_structured_output(self, *args: object, **kwargs: object) -> FlakyChatClient:
        del args, kwargs
        return self

    async def ainvoke(self, messages: object) -> AIMessage | dict[str, object]:
        del messages
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise httpx.ReadError("peer closed")
        return AIMessage(content="ok")


@pytest.mark.asyncio
async def test_embedding_model_retries_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("src.shared.app.retry.asyncio.sleep", no_sleep)
    client = FlakyEmbedClient(failures_before_success=2)
    model = LangChainEmbeddingModel(
        client,  # type: ignore[arg-type]
        dimensions=2,
        model_name="test-embed",
        provider_name="test",
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.01,
            max_delay_seconds=0.01,
        ),
    )

    vectors = await model.embed(["hello"])

    assert vectors == [[0.1, 0.2]]
    assert client.calls == 3


@pytest.mark.asyncio
async def test_chat_complete_retries_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("src.shared.app.retry.asyncio.sleep", no_sleep)
    client = FlakyChatClient(failures_before_success=1)
    model = LangChainChatModel(
        client,  # type: ignore[arg-type]
        model_name="test-chat",
        provider_name="test",
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.01,
            max_delay_seconds=0.01,
        ),
    )

    result = await model.complete(system="s", user="u")

    assert result.content == "ok"
    assert client.calls == 2
