from collections.abc import Sequence

from src.modules.generation.domain.models import ContextChunk


_SYSTEM = (
    "Answer using ONLY the context. Cite evidence as [n]. ",
    "If context is insufficient, say you don't know."
)

class GroundedPromptBuilder:
    def build(self, query: str, contexts: Sequence[ContextChunk]) -> tuple[str, str]:
        block = "\n\n".join(
            f"[{i}] (chunk_id={c.chunk_id})\n{c.content}"
            for i, c in enumerate(contexts, start=1)
        )
        user = f"CONTEXT:\n{block}\n\nQUESTION: {query}"
        return _SYSTEM, user
