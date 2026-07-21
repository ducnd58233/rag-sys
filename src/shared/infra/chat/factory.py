from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.shared.app.ports import IChatModel
from src.shared.app.retry import RetryPolicy
from src.shared.configs.settings import ChatSettings
from src.shared.infra.chat.langchain import LangChainChatModel


class ChatModelFactory:
    @staticmethod
    def from_settings(
        settings: ChatSettings,
        *,
        retry_policy: RetryPolicy,
    ) -> IChatModel:
        match settings.provider.lower():
            case "ollama":
                client = ChatOllama(
                    model=settings.model,
                    base_url=settings.ollama.url,
                    temperature=settings.temperature,
                    num_ctx=settings.num_ctx,
                    num_predict=settings.num_predict,
                    client_kwargs={"timeout": settings.timeout_seconds},
                    async_client_kwargs={"timeout": settings.timeout_seconds},
                )
            case "vllm":
                client = ChatOpenAI(
                    model=settings.model,
                    openai_api_base=f"{settings.vllm.chat_url.rstrip('/')}/v1",
                    openai_api_key="EMPTY",
                    temperature=settings.temperature,
                    request_timeout=settings.timeout_seconds,
                )
            case _:
                raise ValueError(f"Unsupported chat provider: {settings.provider}")
        return LangChainChatModel(
            client,
            model_name=settings.model,
            provider_name=settings.provider.lower(),
            retry_policy=retry_policy,
        )
