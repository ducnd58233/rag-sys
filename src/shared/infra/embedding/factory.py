from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.shared.app.ports import IEmbeddingModel
from src.shared.configs.settings import EmbeddingSettings
from src.shared.infra.embedding.langchain import LangChainEmbeddingModel


class EmbeddingModelFactory:
    @staticmethod
    def from_settings(settings: EmbeddingSettings) -> IEmbeddingModel:
        match settings.provider.lower():
            case "ollama":
                client = OllamaEmbeddings(
                    model=settings.model,
                    base_url=settings.ollama.url,
                )
            case "vllm":
                client = OpenAIEmbeddings(
                    model=settings.model,
                    base_url=f"{settings.vllm.embedding_url.rstrip('/')}/v1",
                    api_key="EMPTY",
                    check_embedding_ctx_length=False,
                    timeout=settings.timeout_seconds,
                )
            case _:
                raise ValueError(f"Unsupported embedding provider: {settings.provider}")

        return LangChainEmbeddingModel(
            client,
            dimensions=settings.dimensions,
            model_name=settings.model,
        )
