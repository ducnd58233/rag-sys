from __future__ import annotations

from collections.abc import Sequence

from src.modules.retrieval.app.dto import RetrievalFilter
from src.modules.retrieval.app.ports import ILexicalSearcher, IRetrievalStrategy
from src.modules.retrieval.domain.models import HitChunk
from src.modules.retrieval.domain.plan import RetrievalPlan
from src.shared.configs.settings import RetrievalSettings


class LexicalStrategy(IRetrievalStrategy):
    def __init__(
        self,
        searcher: ILexicalSearcher,
        settings: RetrievalSettings,
    ) -> None:
        self._searcher = searcher
        self._settings = settings

    @property
    def name(self) -> str:
        return "lexical"

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
        return await self._searcher.search(
            query,
            limit=self._settings.num_candidates,
            filters=filters,
        )
