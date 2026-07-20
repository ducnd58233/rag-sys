from __future__ import annotations

from dataclasses import dataclass

from src.modules.retrieval.app.router import CompositeQueryRouter, LlmRouter, RuleRouter
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
    StructuredStrategy,
)
from src.shared.app.ports import IChatModel, IEmbeddingModel
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
        chat_model: IChatModel | None = None,
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
                StructuredStrategy(
                    elasticsearch,
                    settings.elasticsearch,
                    settings.routing,
                ),
                HybridStrategy(lexical_strategy, semantic_strategy, rank_fusion),
                lexical_strategy,
                semantic_strategy,
            )
        )
        query_router = CompositeQueryRouter(
            settings.routing,
            RuleRouter(settings.routing),
            (
                LlmRouter(
                    settings.routing,
                    chat_model,
                    allowed_strategies=strategy_registry.names,
                )
                if chat_model is not None
                else None
            ),
        )
        return RetrievalComponents(
            retrieve=RetrieveUseCase(
                retrieval_settings=retrieval,
                query_router=query_router,
                strategy_registry=strategy_registry,
                rank_fusion=rank_fusion,
            ),
        )
