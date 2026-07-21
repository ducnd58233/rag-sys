from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from src.shared.app.ports import ChatResult, IChatModel, IEmbeddingModel
from src.shared.app.retry import (
    RetryPolicy,
    is_transient_io_error,
    run_with_retry,
)

TSchema = TypeVar("TSchema", bound=BaseModel)


class RetryingEmbeddingModel:
    """Decorator that retries transient I/O failures around embed calls."""

    def __init__(
        self,
        inner: IEmbeddingModel,
        *,
        policy: RetryPolicy,
    ) -> None:
        self._inner = inner
        self._policy = policy

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await run_with_retry(
            lambda: self._inner.embed(texts),
            policy=self._policy,
            is_retryable=is_transient_io_error,
            operation_name="embedding",
        )


class RetryingChatModel:
    """Decorator that retries transient I/O failures around chat calls."""

    def __init__(
        self,
        inner: IChatModel,
        *,
        policy: RetryPolicy,
    ) -> None:
        self._inner = inner
        self._policy = policy

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        return await run_with_retry(
            lambda: self._inner.complete(
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            policy=self._policy,
            is_retryable=is_transient_io_error,
            operation_name="chat",
        )

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[TSchema],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> TSchema:
        return await run_with_retry(
            lambda: self._inner.complete_structured(
                system=system,
                user=user,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            policy=self._policy,
            is_retryable=is_transient_io_error,
            operation_name="chat_structured",
        )

    def bind_tools(self, tools: Sequence[object]) -> IChatModel:
        return RetryingChatModel(
            self._inner.bind_tools(tools),
            policy=self._policy,
        )
