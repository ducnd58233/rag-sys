from __future__ import annotations

from pydantic import BaseModel, Field

from src.shared.app.ports import IChatModel

_SYSTEM = """
<responsibility>
You extract graph search anchors from one retrieval query for a knowledge-graph
traversal. Return only entity names to match against the graph - never answer the
query.
</responsibility>

<rules>
1. Return short canonical entity names, not full phrases or sentences.
2. Include every distinct entity the query asks about, not just the first one.
3. If the query does not name any concrete entity, return an empty list rather than
   guessing one.
</rules>

<examples>
<example>
<query>What does the checkout service depend on?</query>
<output>{"entities": ["checkout service"]}</output>
</example>
<example>
<query>How are deploy-1832 and the payment gateway timeout incident related?</query>
<output>{"entities": ["deploy-1832", "payment gateway timeout incident"]}</output>
</example>
<example>
<query>What is our refund policy?</query>
<output>{"entities": []}</output>
</example>
</examples>
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
        )
        return tuple(entity.strip() for entity in result.entities if entity.strip())
