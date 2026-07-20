from __future__ import annotations

from collections.abc import Mapping, Sequence

from neo4j import AsyncGraphDatabase, RoutingControl

from src.shared.configs.settings import GraphDbSettings


class Neo4jGraphDb:
    def __init__(self, settings: GraphDbSettings) -> None:
        self._database = settings.database
        self._driver = AsyncGraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password.get_secret_value()),
        )

    async def initialize(self) -> None:
        statements = (
            "CREATE INDEX document_lookup IF NOT EXISTS FOR (d:Document) ON (d.org_id, d.document_id)",
            "CREATE INDEX version_lookup IF NOT EXISTS FOR (v:Version) ON (v.org_id, v.document_version_id)",
            "CREATE INDEX entity_lookup IF NOT EXISTS FOR (e:Entity) ON (e.org_id, e.name)",
        )
        for statement in statements:
            await self.write(statement)

    async def write(
        self,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> None:
        await self._driver.execute_query(
            statement,
            parameters_=(dict(parameters or {})),
            database_=self._database,
            routing_=RoutingControl.WRITE,
        )

    async def read(
        self,
        statement: str,
        parameters: Mapping[str, object] | None = None,
    ) -> Sequence[Mapping[str, object]]:
        result = await self._driver.execute_query(
            statement,
            parameters_=(dict(parameters or {})),
            database_=self._database,
            routing_=RoutingControl.READ,
        )
        return tuple(record.data() for record in result.records)

    async def close(self) -> None:
        await self._driver.close()
