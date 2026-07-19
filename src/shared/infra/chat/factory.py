from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.shared.app.ports import IChatModel
from src.shared.configs.settings import ChatSettings
from src.shared.infra.chat.langchain import LangChainChatModel


class ChatModelFactory:
    @staticmethod
    def from_settings(settings: ChatSettings) -> IChatModel:
        match settings.provider.lower():
            case "ollama":
                client = ChatOllama(
                    model=settings.model,
                    base_url=settings.ollama.url,
                    temperature=settings.temperature,
                )
            case "vllm":
                client = ChatOpenAI(
                    model=settings.model,
                    base_url=f"{settings.vllm.chat_url.rstrip('/')}/v1",
                    api_key="EMPTY",
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                    timeout=settings.timeout_seconds,
                )
            case _:
                raise ValueError(f"Unsupported chat provider: {settings.provider}")
        return LangChainChatModel(client, model_name=settings.model)
