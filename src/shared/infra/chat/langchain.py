from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel

from src.shared.app.ports import ChatResult, IChatModel, ToolCall
from src.shared.app.retry import (
    RetryPolicy,
    is_transient_io_error,
    run_with_retry,
)
from src.shared.observability.metrics import (
    gen_ai_client_operation_duration,
    gen_ai_client_token_usage,
    llm_tokens,
)

TSchema = TypeVar("TSchema", bound=BaseModel)

_tracer = trace.get_tracer(__name__)


class LangChainChatModel:
    def __init__(
        self,
        client: BaseChatModel,
        *,
        model_name: str,
        provider_name: str,
        retry_policy: RetryPolicy,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._provider_name = provider_name
        self._retry_policy = retry_policy

    def _bound(self, *, temperature: float | None) -> BaseChatModel:
        if temperature is None:
            return self._client
        return self._client.bind(temperature=temperature)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> ChatResult:
        model = self._bound(temperature=temperature)
        metric_attributes = self._metric_attributes("chat")
        span_attributes = {
            **metric_attributes,
            "gen_ai.request.stream": False,
        }
        if temperature is not None:
            span_attributes["gen_ai.request.temperature"] = temperature

        started_at = time.perf_counter()
        with _tracer.start_as_current_span(
            f"chat {self._model_name}",
            attributes=span_attributes,
        ) as span:
            try:
                message = await run_with_retry(
                    lambda: model.ainvoke(
                        [
                            SystemMessage(content=system),
                            HumanMessage(content=user),
                        ],
                    ),
                    policy=self._retry_policy,
                    is_retryable=is_transient_io_error,
                    operation_name="chat",
                )
            except Exception as error:
                error_type = error.__class__.__name__
                span.set_status(Status(StatusCode.ERROR, error_type))
                gen_ai_client_operation_duration.record(
                    time.perf_counter() - started_at,
                    {**metric_attributes, "error.type": error_type},
                )
                raise
            gen_ai_client_operation_duration.record(
                time.perf_counter() - started_at,
                metric_attributes,
            )
            self._record_token_usage(message, operation_name="chat")
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
    ) -> TSchema:
        model = self._bound(temperature=temperature)
        structured = model.with_structured_output(
            schema,
            method="json_schema",
            include_raw=True,
        )
        metric_attributes = self._metric_attributes("chat")
        span_attributes = {
            **metric_attributes,
            "gen_ai.output.type": "json",
            "gen_ai.request.stream": False,
        }
        if temperature is not None:
            span_attributes["gen_ai.request.temperature"] = temperature

        started_at = time.perf_counter()
        with _tracer.start_as_current_span(
            f"chat {self._model_name}",
            attributes=span_attributes,
        ) as span:
            try:

                async def _invoke() -> dict[str, object]:
                    invoked = await structured.ainvoke(
                        [
                            SystemMessage(content=system),
                            HumanMessage(content=user),
                        ],
                    )
                    if invoked["parsing_error"] is not None:
                        raise invoked["parsing_error"]
                    return invoked

                result = await run_with_retry(
                    _invoke,
                    policy=self._retry_policy,
                    is_retryable=is_transient_io_error,
                    operation_name="chat_structured",
                )
            except Exception as error:
                error_type = error.__class__.__name__
                span.set_status(Status(StatusCode.ERROR, error_type))
                gen_ai_client_operation_duration.record(
                    time.perf_counter() - started_at,
                    {**metric_attributes, "error.type": error_type},
                )
                raise
            gen_ai_client_operation_duration.record(
                time.perf_counter() - started_at,
                metric_attributes,
            )
            self._record_token_usage(result["raw"], operation_name="chat")
        parsed = result["parsed"]
        if isinstance(parsed, schema):
            return parsed
        return schema.model_validate(parsed)

    def bind_tools(self, tools: Sequence[object]) -> IChatModel:
        return LangChainChatModel(
            self._client.bind_tools(tools),
            model_name=self._model_name,
            provider_name=self._provider_name,
            retry_policy=self._retry_policy,
        )

    def _metric_attributes(self, operation_name: str) -> dict[str, str]:
        return {
            "gen_ai.operation.name": operation_name,
            "gen_ai.provider.name": self._provider_name,
            "gen_ai.request.model": self._model_name,
        }

    def _record_token_usage(self, message: AIMessage, *, operation_name: str) -> None:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            return
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        span = trace.get_current_span()
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            llm_tokens.add(
                input_tokens,
                {"model": self._model_name, "direction": "input"},
            )
            gen_ai_client_token_usage.record(
                input_tokens,
                {
                    **self._metric_attributes(operation_name),
                    "gen_ai.token.type": "input",
                },
            )
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            llm_tokens.add(
                output_tokens,
                {"model": self._model_name, "direction": "output"},
            )
            gen_ai_client_token_usage.record(
                output_tokens,
                {
                    **self._metric_attributes(operation_name),
                    "gen_ai.token.type": "output",
                },
            )
