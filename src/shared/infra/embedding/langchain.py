from collections.abc import Sequence

from langchain.embeddings import Embeddings
from opentelemetry import trace

_tracer = trace.get_tracer(__name__)


class LangChainEmbeddingModel:
    def __init__(self, client: Embeddings, *, dimensions: int, model_name: str) -> None:
        self._client = client
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with _tracer.start_as_current_span(
            "llm.embed",
            attributes={
                "gen_ai.request.model": self._model_name,
                "document_count": len(texts),
            },
        ):
            vectors = await self._client.aembed_documents(list(texts))
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise RuntimeError(
                    f"Expected dim={self._dimensions}, got dim={len(vector)}"
                )
        return vectors
