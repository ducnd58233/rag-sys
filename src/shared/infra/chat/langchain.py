from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from opentelemetry import trace
from pydantic import BaseModel

from src.shared.app.ports import ChatResult, IChatModel, ToolCall
from src.shared.observability.metrics import llm_tokens

TSchema = TypeVar("TSchema", bound=BaseModel)

_tracer = trace.get_tracer(__name__)


class LangChainChatModel:
    def __init__(self, client: BaseChatModel, *, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

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
        with _tracer.start_as_current_span(
            "llm.chat",
            attributes={"gen_ai.request.model": self._model_name},
        ):
            message = await model.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=user)],
            )
            self._record_token_usage(message)
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
        with _tracer.start_as_current_span(
            "llm.chat",
            attributes={"gen_ai.request.model": self._model_name},
        ):
            result = await structured.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=user)],
            )
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)

    def bind_tools(self, tools: Sequence[object]) -> IChatModel:
        return LangChainChatModel(
            self._client.bind_tools(tools),
            model_name=self._model_name,
        )

    def _record_token_usage(self, message: AIMessage) -> None:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            return
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_tokens is not None:
            llm_tokens.add(
                input_tokens,
                {"model": self._model_name, "direction": "input"},
            )
        if output_tokens is not None:
            llm_tokens.add(
                output_tokens,
                {"model": self._model_name, "direction": "output"},
            )
