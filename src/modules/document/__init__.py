from __future__ import annotations

from dataclasses import dataclass

from src.modules.document.app.ports import (
    IDocumentUnitOfWork,
    IIngestionRequestPublisher,
)
from src.modules.document.app.use_cases.complete_upload import (
    CompleteDocumentUploadUseCase,
)
from src.modules.document.app.use_cases.create_upload_url import (
    CreateDocumentUploadUrlUseCase,
)
from src.shared.app.ports import IIdGenerator, IObjectStorage
from src.shared.configs.settings import DocumentSettings

__all__ = [
    "CompleteDocumentUploadUseCase",
    "CreateDocumentUploadUrlUseCase",
    "DocumentComponentFactory",
    "DocumentComponents",
]


@dataclass(frozen=True, slots=True)
class DocumentComponents:
    create_upload_url: CreateDocumentUploadUrlUseCase
    complete_upload: CompleteDocumentUploadUseCase


class DocumentComponentFactory:
    @staticmethod
    def build(
        *,
        settings: DocumentSettings,
        document_uow: IDocumentUnitOfWork,
        object_storage: IObjectStorage,
        id_generator: IIdGenerator,
        ingestion_publisher: IIngestionRequestPublisher,
    ) -> DocumentComponents:
        return DocumentComponents(
            create_upload_url=CreateDocumentUploadUrlUseCase(
                document_uow=document_uow,
                object_storage=object_storage,
                id_generator=id_generator,
                allowed_mime_types=settings.allowed_mime_types,
                max_upload_size_bytes=settings.max_upload_size_bytes,
                upload_url_ttl_seconds=settings.upload_url_ttl_seconds,
            ),
            complete_upload=CompleteDocumentUploadUseCase(
                document_uow=document_uow,
                object_storage=object_storage,
                ingestion_publisher=ingestion_publisher,
            ),
        )
