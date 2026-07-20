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
    # Function-scoped, not session-scoped: pytest-asyncio gives each test
    # its own event loop by default, and an AsyncEngine's connections are
    # bound to the loop that created them. A session-scoped engine here
    # would work in test 1 and then raise "Event loop is closed" in test
    # 2 (this was tried and reverted). Reusing one engine across tests
    # would need a session-scoped event loop too, which is a bigger,
    # riskier change than the connection-pool overhead it would save.
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


@pytest_asyncio.fixture(scope="session")
async def _graphdb_initialized(graphdb_settings: GraphDbSettings) -> None:
    # Index creation only needs to happen once against the session-scoped
    # container, not once per test.
    db = Neo4jGraphDb(graphdb_settings)
    try:
        await db.initialize()
    finally:
        await db.close()


@pytest_asyncio.fixture
async def graphdb(
    graphdb_settings: GraphDbSettings,
    _graphdb_initialized: None,
) -> AsyncIterator[Neo4jGraphDb]:
    db = Neo4jGraphDb(graphdb_settings)
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture(scope="session")
def _id_block_counter() -> Iterator[int]:
    # The containers are session-scoped and never reset between tests, so
    # every test needs its own org_id/document_id/etc. space. Each test
    # gets a 1,000-id block (unique_id, unique_id + 1, ... up to + 999) so
    # a test using several related ids internally can never collide with
    # another test's block. This only holds under sequential execution:
    # each pytest-xdist worker would get its own counter starting back at
    # 100, silently reintroducing collisions across workers, so refuse to
    # run under more than one xdist worker rather than fail confusingly.
    worker_count = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
    if worker_count is not None and int(worker_count) > 1:
        raise RuntimeError(
            "tests/integration cannot run under pytest-xdist with more "
            "than one worker: unique_id allocation is per-process and "
            "would collide across workers. Run with -n0 or without -n."
        )
    return itertools.count(start=100)


@pytest.fixture
def unique_id(_id_block_counter: Iterator[int]) -> int:
    return next(_id_block_counter) * 1_000
