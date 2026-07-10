from src.shared.configs.settings import ElasticsearchSettings
from elasticsearch import AsyncElasticsearch

class Elasticsearch:
    def __init__(self, setting: ElasticsearchSettings):
        self._client = AsyncElasticsearch(
            hosts=setting.urls,
            request_timeout=setting.request_timeout_seconds,
        )

    async def ping(self) -> bool:
        return await self._client.ping()

    async def close(self) -> None:
        await self._client.close()