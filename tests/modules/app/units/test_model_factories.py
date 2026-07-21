from __future__ import annotations

from src.shared.app.retry import RetryPolicy
from src.shared.configs.settings import ChatSettings, EmbeddingSettings, VLLMSettings
from src.shared.infra.chat.factory import ChatModelFactory
from src.shared.infra.embedding.factory import EmbeddingModelFactory


def test_vllm_embedding_factory_uses_openai_compatible_field_names() -> None:
    settings = EmbeddingSettings(
        provider="vllm",
        model="qwen3-embedding",
        timeout_seconds=12.5,
        vllm=VLLMSettings(embedding_url="http://localhost:8001"),
    )

    model = EmbeddingModelFactory.from_settings(
        settings,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    client = model._client  # noqa: SLF001

    assert client.openai_api_base == "http://localhost:8001/v1"
    assert client.request_timeout == 12.5
    assert client.check_embedding_ctx_length is False
    assert client.tiktoken_enabled is False
    assert client.openai_api_key.get_secret_value() == "EMPTY"


def test_vllm_chat_factory_uses_openai_compatible_field_names() -> None:
    settings = ChatSettings(
        provider="vllm",
        model="qwen2.5",
        timeout_seconds=9.0,
        vllm=VLLMSettings(chat_url="http://localhost:8003"),
    )

    model = ChatModelFactory.from_settings(
        settings,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    client = model._client  # noqa: SLF001

    assert client.openai_api_base == "http://localhost:8003/v1"
    assert client.request_timeout == 9.0
    assert client.openai_api_key.get_secret_value() == "EMPTY"


def test_ollama_chat_factory_passes_num_ctx_and_num_predict() -> None:
    settings = ChatSettings(
        provider="ollama",
        model="qwen3:4b-instruct",
        num_ctx=8192,
        num_predict=2048,
    )

    model = ChatModelFactory.from_settings(
        settings,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    client = model._client  # noqa: SLF001

    assert client.num_ctx == 8192
    assert client.num_predict == 2048
