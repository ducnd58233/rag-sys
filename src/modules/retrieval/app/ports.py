from collections.abc import Sequence
from typing import Protocol

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import RetrievalPlan


class ILexicalSearcher(Protocol):
    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]: ...


class IDenseSearcher(Protocol):
    async def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        num_candidates: int,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]: ...


class IRankFusion(Protocol):
    def fuse(
        self,
        ranked_lists: Sequence[Sequence[HitChunk]],
        *,
        top_k: int,
        weights: Sequence[float] | None = None,
    ) -> tuple[HitChunk, ...]: ...


class IRetrievalStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def supports(self, plan: RetrievalPlan) -> bool: ...

    async def retrieve(
        self,
        query: str,
        *,
        plan: RetrievalPlan,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]: ...


class IGraphSearcher(Protocol):
    async def related_document_ids(
        self,
        *,
        org_id: int,
        entities: Sequence[str],
        max_hops: int,
        limit: int,
    ) -> Sequence[str]: ...


class IGraphQueryAnalyzer(Protocol):
    async def analyze(self, query: str) -> Sequence[str]: ...


class IQueryRouter(Protocol):
    async def route(
        self,
        query: str,
        *,
        filters: RetrievalFilter,
        top_k: int,
    ) -> RetrievalPlan: ...
