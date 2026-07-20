from __future__ import annotations

import time
from dataclasses import dataclass
from types import TracebackType

import aiohttp


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str | None
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    items: tuple[RetrievedChunk, ...]
    latency_ms: float
    router_kind: str
    router_reason: str
    strategies: tuple[str, ...]


# Talks to the running app over HTTP rather than importing RetrieveUseCase directly,
# so a measured run includes real serialization, middleware, and the actual
# composition root - the same thing an actual caller would see.
class EvalHttpClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> EvalHttpClient:
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._session is not None
        await self._session.close()

    async def retrieve(
        self,
        *,
        org_id: int,
        query: str,
        top_k: int,
    ) -> RetrievalResponse:
        assert (
            self._session is not None
        ), "use EvalHttpClient as an async context manager"
        started_at = time.perf_counter()
        async with self._session.post(
            f"{self._base_url}/api/v1/retrieval/search",
            json={"org_id": org_id, "query": query, "top_k": top_k},
        ) as response:
            response.raise_for_status()
            payload = await response.json()
        latency_ms = (time.perf_counter() - started_at) * 1000

        items = tuple(
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                filename=item.get("metadata", {}).get("filename"),
                score=item["score"],
            )
            for item in payload["items"]
        )
        return RetrievalResponse(
            items=items,
            latency_ms=latency_ms,
            router_kind=payload["plan"]["router_kind"],
            router_reason=payload["plan"]["reason"],
            strategies=tuple(
                selection["name"] for selection in payload["plan"]["strategies"]
            ),
        )
