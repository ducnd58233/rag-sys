from typing import Sequence
import aiohttp
from src.shared.configs.settings import EmbeddingSettings
import logging

logger = logging.getLogger(__name__)

class VllmEmbedding:
    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        
    @property
    def dimensions(self) -> int:
        return self._settings.dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        base_url = self._settings.vllm.embedding_url.rstrip("/")
        endpoint = f"{base_url}/v1/embeddings"
        timeout = aiohttp.ClientTimeout(total=self._settings.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    endpoint,
                    json={
                        "model": self._settings.model,
                        "input": list(texts),
                        "encoding_format": "float",
                    },
                ) as response:
                    response.raise_for_status()
                    body = await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Failed to embed texts: {e}")
            raise e
        except Exception as e:
            logger.error(f"Timeout embedding texts: {e}")
            raise e

        data = body.get("data", [])
        if not data:
            raise RuntimeError("No embeddings found in response")
        
        if len(data) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(data)}")

        sorted_items = sorted(data, key=lambda item: int(item["index"]))
        vectors = [list(map(float, item["embedding"])) for item in sorted_items]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}")

        for vector in vectors:
            if len(vector) != self.dimensions:
                raise RuntimeError(f"Expected dim={self.dimensions}, got dim={len(vector)}")

        return vectors