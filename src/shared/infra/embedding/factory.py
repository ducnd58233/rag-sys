from src.shared.configs.settings import EmbeddingSettings
from src.shared.infra.embedding.ollama import OllamaEmbedding
from src.shared.infra.embedding.ports import IEmbeddingModel
from src.shared.infra.embedding.vllm import VllmEmbedding


class EmbeddingModelFactory:
    @staticmethod
    def from_settings(settings: EmbeddingSettings) -> IEmbeddingModel:
        match settings.provider.lower():
            case "ollama":
                return OllamaEmbedding(settings)
            case "vllm":
                return VllmEmbedding(settings)
            case _:
                raise ValueError(f"Unsupported embedding provider: {settings.provider}")