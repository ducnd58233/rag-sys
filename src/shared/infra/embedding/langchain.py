from collections.abc import Sequence
from langchain_community.embeddings import Embeddings


class LangChainEmbeddingModel:
    def __init__(self, client: Embeddings, *, dimensions: int) -> None:
        self._client = client
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await self._client.aembed_documents(list(texts))
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise RuntimeError(
                    f"Expected dim={self._dimensions}, got dim={len(vector)}"
                )
        return vectors