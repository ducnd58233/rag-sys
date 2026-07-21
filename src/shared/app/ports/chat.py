from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

TSchema = TypeVar("TSchema", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, object]


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


class IChatModel(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> ChatResult: ...

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[TSchema],
        temperature: float | None = None,
    ) -> TSchema: ...

    def bind_tools(self, tools: Sequence[object]) -> IChatModel: ...
