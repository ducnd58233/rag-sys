from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from src.modules.document.app.ports import IDocumentTransaction, ICaseRepository, IDocumentRepository, IDocumentUnitOfWork, IDocumentVersionRepository, IStoredObjectRepository
from src.modules.document.infra.persistence.repo import SqlAlchemyCaseRepository,SqlAlchemyDocumentRepository,SqlAlchemyDocumentVersionRepository,SqlAlchemyStoredObjectRepository


@dataclass(frozen=True, slots=True)
class SqlAlchemyDocumentTransaction(IDocumentTransaction):
    cases: ICaseRepository
    documents: IDocumentRepository
    document_versions: IDocumentVersionRepository
    stored_objects: IStoredObjectRepository

class SqlAlchemyDocumentUnitOfWork(IDocumentUnitOfWork):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
    
    @asynccontextmanager
    async def begin(self) -> AsyncIterator[SqlAlchemyDocumentTransaction]:
        async with self._session_factory() as session:
            async with session.begin():
                yield SqlAlchemyDocumentTransaction(
                    cases=SqlAlchemyCaseRepository(session),
                    documents=SqlAlchemyDocumentRepository(session),
                    document_versions=SqlAlchemyDocumentVersionRepository(session),
                    stored_objects=SqlAlchemyStoredObjectRepository(session),
                )