import logging
from typing import Sequence
import aiohttp
from src.shared.configs.settings import EmbeddingSettings

logger = logging.getLogger(__name__)

class OllamaEmbedding:
    def __init__(self, settings: EmbeddingSettings):
        self._settings = settings
    
    @property
    def dimensions(self) -> int:
        return self._settings.dimensions

    async def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        base_url = self._settings.ollama.url.rstrip("/")
        endpoint = f"{base_url}/api/embed"
        timeout = aiohttp.ClientTimeout(total=self._settings.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    endpoint,
                        json={
                        "model": self._settings.model,
                        "input": list(texts),
                    },
                ) as response:
                    response.raise_for_status()
                    body = await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"HTTP status error embedding text: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise e

        raw_embeddings = body.get("embeddings", [])
        if not raw_embeddings:
            raise RuntimeError("No embeddings found in response")

        vectors = [list(map(float, vector)) for vector in raw_embeddings]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}")

        for vector in vectors:
            if len(vector) != self.dimensions:
                raise RuntimeError(f"Expected dim={self.dimensions}, got dim={len(vector)}")

        return vectors
