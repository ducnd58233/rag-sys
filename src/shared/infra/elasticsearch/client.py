from elasticsearch import AsyncElasticsearch

from src.shared.configs.settings import ElasticsearchSettings


class Elasticsearch:
    def __init__(self, settings: ElasticsearchSettings):
        self._client = AsyncElasticsearch(
            hosts=settings.urls,
            request_timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,
            retry_on_timeout=settings.retry_on_timeout,
            retry_backoff_base=settings.retry_backoff_base,
            retry_backoff_cap=settings.retry_backoff_cap,
        )

    @property
    def client(self) -> AsyncElasticsearch:
        return self._client

    async def ping(self) -> bool:
        return await self._client.ping()

    async def close(self) -> None:
        await self._client.close()
