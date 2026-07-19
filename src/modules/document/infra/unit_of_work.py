from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.modules.document.app.ports import (
    IDocumentRepository,
    IDocumentTransaction,
    IDocumentUnitOfWork,
    IDocumentVersionRepository,
    IStoredObjectRepository,
)
from src.modules.document.infra.persistence.repo import (
    SqlAlchemyDocumentRepository,
    SqlAlchemyDocumentVersionRepository,
    SqlAlchemyStoredObjectRepository,
)


@dataclass(frozen=True, slots=True)
class SqlAlchemyDocumentTransaction(IDocumentTransaction):
    _session: AsyncSession
    documents: IDocumentRepository
    document_versions: IDocumentVersionRepository
    stored_objects: IStoredObjectRepository

    async def flush(self) -> None:
        await self._session.flush()


class SqlAlchemyDocumentUnitOfWork(IDocumentUnitOfWork):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[SqlAlchemyDocumentTransaction]:
        async with self._session_factory() as session:
            async with session.begin():
                yield SqlAlchemyDocumentTransaction(
                    _session=session,
                    documents=SqlAlchemyDocumentRepository(session),
                    document_versions=SqlAlchemyDocumentVersionRepository(session),
                    stored_objects=SqlAlchemyStoredObjectRepository(session),
                )
