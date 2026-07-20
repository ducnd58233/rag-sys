from __future__ import annotations

from dataclasses import dataclass

from src.modules.retrieval.app.strategy_registry import RetrievalStrategyRegistry
from src.modules.retrieval.app.use_cases.retrieve import RetrieveUseCase
from src.modules.retrieval.infra.elasticsearch.dense_searcher import (
    ElasticsearchDenseSearcher,
)
from src.modules.retrieval.infra.elasticsearch.lexical_searcher import (
    ElasticsearchLexicalSearcher,
)
from src.modules.retrieval.infra.fusion.reciprocal_rank import ReciprocalRankFusion
from src.modules.retrieval.infra.strategies import (
    HybridStrategy,
    LexicalStrategy,
    SemanticStrategy,
)
from src.shared.app.ports import IEmbeddingModel
from src.shared.configs.settings import Settings
from src.shared.infra.elasticsearch.client import Elasticsearch

__all__ = ["RetrievalComponentFactory", "RetrievalComponents", "RetrieveUseCase"]


@dataclass(frozen=True, slots=True)
class RetrievalComponents:
    retrieve: RetrieveUseCase


class RetrievalComponentFactory:
    @staticmethod
    def build(
        *,
        settings: Settings,
        elasticsearch: Elasticsearch,
        embedder: IEmbeddingModel,
    ) -> RetrievalComponents:
        retrieval = settings.retrieval
        rank_fusion = ReciprocalRankFusion(rank_constant=retrieval.rank_constant)
        lexical_strategy = LexicalStrategy(
            ElasticsearchLexicalSearcher(elasticsearch, settings.elasticsearch),
            retrieval,
        )
        semantic_strategy = SemanticStrategy(
            embedder,
            ElasticsearchDenseSearcher(elasticsearch, settings.elasticsearch),
            retrieval,
        )
        strategy_registry = RetrievalStrategyRegistry(
            (
                HybridStrategy(lexical_strategy, semantic_strategy, rank_fusion),
                lexical_strategy,
                semantic_strategy,
            )
        )
        return RetrievalComponents(
            retrieve=RetrieveUseCase(
                retrieval_settings=retrieval,
                strategy_registry=strategy_registry,
            ),
        )
