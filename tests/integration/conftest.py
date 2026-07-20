from __future__ import annotations

import itertools
import os
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from testcontainers.elasticsearch import ElasticSearchContainer
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer

from src.shared.configs.settings import (
    DatabaseSettings,
    ElasticsearchSettings,
    GraphDbSettings,
)
from src.shared.infra.database import Database
from src.shared.infra.elasticsearch.client import Elasticsearch
from src.shared.infra.graphdb import Neo4jGraphDb

# Pinned to the same image versions as deployments/docker/docker-compose.yml,
# so integration tests exercise the same server versions used in dev/prod,
# not whatever "latest" happens to resolve to on a given day.
POSTGRES_IMAGE = "postgres:18.0-bookworm"
ELASTICSEARCH_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:9.4.3"
NEO4J_IMAGE = "neo4j:2026.06.0"


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[PostgresContainer]:
    with PostgresContainer(POSTGRES_IMAGE, driver="asyncpg") as container:
        yield container


@pytest.fixture(scope="session")
def elasticsearch_container() -> Iterator[ElasticSearchContainer]:
    with ElasticSearchContainer(ELASTICSEARCH_IMAGE) as container:
        yield container


@pytest.fixture(scope="session")
def neo4j_container() -> Iterator[Neo4jContainer]:
    with Neo4jContainer(NEO4J_IMAGE, password="test-password") as container:
        yield container


@pytest.fixture(scope="session")
def database_settings(postgres_container: PostgresContainer) -> DatabaseSettings:
    return DatabaseSettings(url=postgres_container.get_connection_url())


@pytest.fixture(scope="session")
def elasticsearch_settings(
    elasticsearch_container: ElasticSearchContainer,
) -> ElasticsearchSettings:
    host = elasticsearch_container.get_container_host_ip()
    port = elasticsearch_container.get_exposed_port(elasticsearch_container.port)
    return ElasticsearchSettings(urls=[f"http://{host}:{port}"])


@pytest.fixture(scope="session")
def graphdb_settings(neo4j_container: Neo4jContainer) -> GraphDbSettings:
    return GraphDbSettings(
        uri=neo4j_container.get_connection_url(),
        username=neo4j_container.username,
        password=neo4j_container.password,
    )


@pytest.fixture(scope="session")
def _migrated_database_url(database_settings: DatabaseSettings) -> str:
    """Run the project's real Alembic migrations against the container once
    per session, so these tests validate the actual schema shipped to
    production, not a hand-rolled test schema that could drift from it."""
    url = database_settings.url.get_secret_value()
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    try:
        config = Config("alembic.ini")
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
    return url


@pytest_asyncio.fixture
async def database(
    database_settings: DatabaseSettings,
    _migrated_database_url: str,
) -> AsyncIterator[Database]:
    db = Database(database_settings)
    try:
        yield db
    finally:
        await db.close()


@pytest_asyncio.fixture
async def elasticsearch(
    elasticsearch_settings: ElasticsearchSettings,
) -> AsyncIterator[Elasticsearch]:
    es = Elasticsearch(elasticsearch_settings)
    try:
        yield es
    finally:
        await es.close()


@pytest_asyncio.fixture
async def graphdb(graphdb_settings: GraphDbSettings) -> AsyncIterator[Neo4jGraphDb]:
    db = Neo4jGraphDb(graphdb_settings)
    try:
        await db.initialize()
        yield db
    finally:
        await db.close()


@pytest.fixture(scope="session")
def _id_block_counter() -> Iterator[int]:
    # The containers are session-scoped and never reset between tests, so
    # every test needs its own org_id/document_id/etc. space. Each test
    # gets a 1,000-id block (unique_id, unique_id + 1, ... up to + 999) so
    # a test using several related ids internally can never collide with
    # another test's block, even though tests run sequentially and not
    # concurrently in this suite.
    return itertools.count(start=100)


@pytest.fixture
def unique_id(_id_block_counter: Iterator[int]) -> int:
    return next(_id_block_counter) * 1_000
