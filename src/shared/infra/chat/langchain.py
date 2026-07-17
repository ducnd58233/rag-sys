from collections.abc import Sequence
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from src.shared.app.ports import ChatResult, IChatModel, ToolCall

class LangChainChatModel:
    def __init__(self, client: BaseChatModel) -> None:
        self._client = client

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        model = self._client
        bind_kwargs: dict[str, object] = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        if bind_kwargs:
            model = model.bind(**bind_kwargs)

        message = await model.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
        )
        content = message.content if  isinstance(message.content, str) else str(message.content)
        tool_calls = tuple(
            ToolCall(
                id=str(tc.get("id", "")),
                name=str(tc.get("name", "")),
                args=dict(tc.get("args", {})),
            )
            for tc in (message.tool_calls or [])
        )
        return ChatResult(content=content, tool_calls=tool_calls)

    def bind_tools(self, tools: Sequence[object]) -> IChatModel:
        return LangChainChatModel(self._client.bind_tools(tools))