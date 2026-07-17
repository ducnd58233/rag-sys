from __future__ import annotations

from src.modules.retrieval.app.use_cases.retrieve import RetrieveUseCase
from src.modules.retrieval.infra.elasticsearch.dense_searcher import (
    ElasticsearchDenseSearcher,
)
from src.modules.retrieval.infra.elasticsearch.lexical_searcher import (
    ElasticsearchLexicalSearcher,
)
from src.modules.retrieval.infra.fusion.reciprocal_rank import ReciprocalRankFusion
from src.shared.app.ports import IEmbeddingModel
from src.shared.configs.settings import Settings
from src.shared.infra.elasticsearch.client import Elasticsearch

__all__ = ["RetrievalComponentFactory", "RetrieveUseCase"]


class RetrievalComponentFactory:
    @staticmethod
    def build_retrieve_use_case(
        settings: Settings,
        elasticsearch: Elasticsearch,
        embedder: IEmbeddingModel,
    ) -> RetrieveUseCase:
        retrieval = settings.retrieval
        return RetrieveUseCase(
            embedder=embedder,
            retrieval_settings=retrieval,
            lexical_searcher=ElasticsearchLexicalSearcher(
                elasticsearch, settings.elasticsearch
            ),
            dense_searcher=ElasticsearchDenseSearcher(
                elasticsearch, settings.elasticsearch
            ),
            rank_fusion=ReciprocalRankFusion(rank_constant=retrieval.rank_constant),
        )