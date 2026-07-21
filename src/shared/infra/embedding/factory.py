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
                    client_kwargs={"timeout": settings.timeout_seconds},
                    async_client_kwargs={"timeout": settings.timeout_seconds},
                )
            case "vllm":
                # langchain-openai stores these as openai_api_base / openai_api_key /
                # request_timeout (aliases: base_url, api_key, timeout). Prefer the
                # canonical names so type checkers and stubs agree with the model.
                # https://reference.langchain.com/python/langchain-openai/embeddings/base/OpenAIEmbeddings
                client = OpenAIEmbeddings(
                    model=settings.model,
                    openai_api_base=(
                        f"{settings.vllm.embedding_url.rstrip('/')}/v1"
                    ),
                    openai_api_key="EMPTY",
                    check_embedding_ctx_length=False,
                    tiktoken_enabled=False,
                    request_timeout=settings.timeout_seconds,
                )
            case _:
                raise ValueError(f"Unsupported embedding provider: {settings.provider}")

        return LangChainEmbeddingModel(
            client,
            dimensions=settings.dimensions,
            model_name=settings.model,
            provider_name=settings.provider.lower(),
        )
