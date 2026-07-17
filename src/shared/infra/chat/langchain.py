from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.shared.app.ports import ChatResult, IChatModel, ToolCall

TSchema = TypeVar("TSchema", bound=BaseModel)


class LangChainChatModel:
    def __init__(self, client: BaseChatModel) -> None:
        self._client = client

    def _bound(
        self,
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> BaseChatModel:
        model = self._client
        bind_kwargs: dict[str, object] = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        if bind_kwargs:
            model = model.bind(**bind_kwargs)
        return model

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        model = self._bound(temperature=temperature, max_tokens=max_tokens)
        message = await model.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
        )
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        tool_calls = tuple(
            ToolCall(
                id=str(tc.get("id", "")),
                name=str(tc.get("name", "")),
                args=dict(tc.get("args", {})),
            )
            for tc in (message.tool_calls or [])
        )
        return ChatResult(content=content, tool_calls=tool_calls)

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[TSchema],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> TSchema:
        model = self._bound(temperature=temperature, max_tokens=max_tokens)
        structured = model.with_structured_output(schema, method="json_schema")
        result = await structured.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
        )
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)

    def bind_tools(self, tools: Sequence[object]) -> IChatModel:
        return LangChainChatModel(self._client.bind_tools(tools))