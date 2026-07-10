from src.shared.configs.settings import ElasticsearchSettings
from elasticsearch import AsyncElasticsearch

class Elasticsearch:
    def __init__(self, settings: ElasticsearchSettings):
        self._client = AsyncElasticsearch(
            hosts=settings.urls,
            request_timeout=settings.request_timeout_seconds,
        )

    async def ping(self) -> bool:
        return await self._client.ping()

    async def close(self) -> None:
        await self._client.close()