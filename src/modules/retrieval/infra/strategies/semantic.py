from __future__ import annotations

from collections.abc import Sequence

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.ports import IDenseSearcher, IRetrievalStrategy
from src.modules.retrieval.domain.errors import RetrievalValidationError
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import RetrievalPlan
from src.shared.app.ports import IEmbeddingModel
from src.shared.configs.settings import RetrievalSettings


class SemanticStrategy(IRetrievalStrategy):
    def __init__(
        self,
        embedder: IEmbeddingModel,
        searcher: IDenseSearcher,
        settings: RetrievalSettings,
    ) -> None:
        self._embedder = embedder
        self._searcher = searcher
        self._settings = settings

    @property
    def name(self) -> str:
        return "semantic"

    def supports(self, plan: RetrievalPlan) -> bool:
        return plan.selection_for(self.name) is not None

    async def retrieve(
        self,
        query: str,
        *,
        plan: RetrievalPlan,
        filters: RetrievalFilter,
    ) -> Sequence[HitChunk]:
        if not self.supports(plan):
            return ()
        vectors = await self._embedder.embed([query])
        if not vectors or not vectors[0]:
            raise RetrievalValidationError("failed to embed query")
        return await self._searcher.search(
            vectors[0],
            limit=self._settings.num_candidates,
            num_candidates=self._settings.candidate_k,
            filters=filters,
        )
