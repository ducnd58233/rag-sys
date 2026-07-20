from __future__ import annotations

from pydantic import BaseModel, Field

from src.shared.app.ports import IChatModel

_SYSTEM = """
Extract graph search anchors from a user retrieval query.
Return short entity names that should be matched against a knowledge graph.
Return no answer text.
""".strip()


class GraphQuerySchema(BaseModel):
    entities: list[str] = Field(default_factory=list)


class LlmGraphQueryAnalyzer:
    def __init__(self, chat_model: IChatModel) -> None:
        self._chat = chat_model

    async def analyze(self, query: str) -> tuple[str, ...]:
        result = await self._chat.complete_structured(
            system=_SYSTEM,
            user=f"QUERY: {query}",
            schema=GraphQuerySchema,
            temperature=0.0,
            max_tokens=256,
        )
        return tuple(entity.strip() for entity in result.entities if entity.strip())
