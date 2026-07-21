from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.shared.configs.settings import DatabaseSettings, ElasticsearchSettings
from src.shared.infra.database.client import Database
from src.shared.infra.elasticsearch.client import Elasticsearch


def test_elasticsearch_client_passes_retry_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_async_elasticsearch(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        "src.shared.infra.elasticsearch.client.AsyncElasticsearch",
        fake_async_elasticsearch,
    )

    settings = ElasticsearchSettings(
        urls=["http://localhost:9200"],
        request_timeout_seconds=30.0,
        max_retries=5,
        retry_on_timeout=True,
        retry_backoff_base=1.0,
        retry_backoff_cap=20.0,
    )
    Elasticsearch(settings)

    assert captured["hosts"] == ["http://localhost:9200"]
    assert captured["request_timeout"] == 30.0
    assert captured["max_retries"] == 5
    assert captured["retry_on_timeout"] is True
    assert captured["retry_backoff_base"] == 1.0
    assert captured["retry_backoff_cap"] == 20.0


def test_database_engine_enables_pool_pre_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_create_async_engine(url: str, **kwargs: object) -> MagicMock:
        captured["url"] = url
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        "src.shared.infra.database.client.create_async_engine",
        fake_create_async_engine,
    )
    monkeypatch.setattr(
        "src.shared.infra.database.client.async_sessionmaker",
        lambda **_kwargs: MagicMock(),
    )

    settings = DatabaseSettings(
        pool_size=8,
        max_overflow=4,
        pool_pre_ping=True,
        pool_recycle_seconds=1800,
    )
    Database(settings)

    assert captured["pool_size"] == 8
    assert captured["max_overflow"] == 4
    assert captured["pool_pre_ping"] is True
    assert captured["pool_recycle"] == 1800
