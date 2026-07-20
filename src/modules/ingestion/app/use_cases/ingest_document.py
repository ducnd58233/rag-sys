from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from opentelemetry import trace

from src.modules.document.app.ports import IDocumentUnitOfWork
from src.modules.document.domain.models import (
    DocumentProcessingStatus,
    DocumentVersionRecord,
    StoredObjectRecord,
    StoredObjectStatus,
)
from src.modules.ingestion.app.dto import IngestDocumentRequest, IngestDocumentResult
from src.modules.ingestion.app.ports import (
    IDocumentProcessor,
    ISourceResolver,
    IVectorStore,
)
from src.modules.ingestion.domain.errors import (
    IngestionConflictError,
    IngestionInternalError,
    IngestionNotFoundError,
)
from src.modules.ingestion.domain.models import (
    DocumentId,
    DocumentVersionSource,
    IngestionStatus,
)
from src.shared.app.ports import IEmbeddingModel
from src.shared.observability.metrics import (
    ingestion_document_size,
    ingestion_step_duration,
)

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


class IngestDocumentUseCase:
    def __init__(
        self,
        doc_uow: IDocumentUnitOfWork,
        source_resolver: ISourceResolver,
        processor: IDocumentProcessor,
        embedder: IEmbeddingModel,
        vector_store: IVectorStore,
    ) -> None:
        self._doc_uow = doc_uow
        self._source_resolver = source_resolver
        self._processor = processor
        self._embedder = embedder
        self._vector_store = vector_store

    async def execute(
        self,
        request: IngestDocumentRequest,
    ) -> IngestDocumentResult:
        source = await self._load_source_and_mark_parsing(request)

        if source.processing_status is DocumentProcessingStatus.INDEXED:
            logger.info(
                "Document version already indexed; skipping re-ingestion",
                extra={"document_version_id": source.document_version_id},
            )
            return IngestDocumentResult(
                document_id=source.document_id.value,
                document_version_id=source.document_version_id,
                version_no=source.version_no,
                chunk_count=0,
                status=IngestionStatus.COMPLETED,
            )

        with _tracer.start_as_current_span("ingestion.execute") as span:
            span.set_attribute("document_version_id", source.document_version_id)
            span.set_attribute("org_id", source.org_id)
            ingestion_document_size.record(
                source.size_bytes,
                {"mime_type": source.mime_type},
            )

            try:
                async with self._step("resolve_and_process"):
                    async with self._source_resolver.open(
                        source,
                    ) as document_source:
                        processed = await self._processor.process(
                            document_source,
                        )

                chunks = processed.chunks
                async with self._step("embed"):
                    vectors = (
                        await self._embedder.embed(
                            [chunk.content for chunk in chunks],
                        )
                        if chunks
                        else []
                    )

                if len(vectors) != len(chunks):
                    raise IngestionInternalError(
                        message=("Embedding count does not match chunk count"),
                    )

                metadata = {
                    **processed.metadata,
                    "org_id": str(source.org_id),
                    "document_id": str(source.document_id.value),
                    "document_version_id": str(
                        source.document_version_id,
                    ),
                    "version": str(source.version_no),
                    "filename": source.filename,
                    "mime_type": source.mime_type,
                }

                async with self._step("index"):
                    await self._vector_store.create_index_if_not_exists()
                    await self._vector_store.close_superseded_versions(
                        org_id=source.org_id,
                        document_id=source.document_id,
                        active_document_version_id=source.document_version_id,
                        valid_to=source.valid_from,
                    )
                    await self._vector_store.delete_by_document_version_id(
                        org_id=source.org_id,
                        document_version_id=source.document_version_id,
                    )

                    if chunks:
                        await self._vector_store.upsert(
                            org_id=source.org_id,
                            document_id=source.document_id,
                            document_version_id=source.document_version_id,
                            version_no=source.version_no,
                            metadata=metadata,
                            chunks=chunks,
                            vectors=vectors,
                            valid_from=source.valid_from,
                            valid_to=source.valid_to,
                        )

                await self._mark_indexed(request)
            except Exception:
                await self._mark_failed(request)
                raise

        return IngestDocumentResult(
            document_id=source.document_id.value,
            document_version_id=source.document_version_id,
            version_no=source.version_no,
            chunk_count=len(chunks),
            status=IngestionStatus.COMPLETED,
        )

    @asynccontextmanager
    async def _step(self, name: str) -> AsyncIterator[None]:
        started_at = time.perf_counter()
        outcome = "success"
        with _tracer.start_as_current_span(f"ingestion.{name}"):
            try:
                yield
            except Exception:
                outcome = "failed"
                raise
            finally:
                ingestion_step_duration.record(
                    time.perf_counter() - started_at,
                    {"step": name, "outcome": outcome},
                )

    async def _load_source_and_mark_parsing(
        self,
        request: IngestDocumentRequest,
    ) -> DocumentVersionSource:
        async with self._doc_uow.begin() as transaction:
            version = await transaction.document_versions.get(
                org_id=request.org_id,
                document_version_id=request.document_version_id,
            )
            if version is None:
                raise IngestionNotFoundError(
                    message="Document version not found",
                )

            document = await transaction.documents.get(
                org_id=request.org_id,
                document_id=version.document_id,
            )
            stored_object = await transaction.stored_objects.get(
                org_id=request.org_id,
                stored_object_id=version.storage_object_id,
            )

            if document is None or stored_object is None:
                raise IngestionNotFoundError(
                    message="Document source is incomplete",
                )

            if version.valid_from is None:
                raise IngestionConflictError(
                    message="Document version is not active",
                )

            if version.processing_status is DocumentProcessingStatus.INDEXED:
                return self._build_source(
                    request,
                    version,
                    stored_object,
                    processing_status=DocumentProcessingStatus.INDEXED,
                )

            if stored_object.status is not StoredObjectStatus.AVAILABLE:
                raise IngestionConflictError(
                    message="Stored object is not available",
                )

            await transaction.document_versions.set_processing_status(
                org_id=request.org_id,
                document_version_id=request.document_version_id,
                status=DocumentProcessingStatus.PARSING,
            )

            return self._build_source(
                request,
                version,
                stored_object,
                processing_status=DocumentProcessingStatus.PARSING,
            )

    def _build_source(
        self,
        request: IngestDocumentRequest,
        version: DocumentVersionRecord,
        stored_object: StoredObjectRecord,
        *,
        processing_status: DocumentProcessingStatus,
    ) -> DocumentVersionSource:
        return DocumentVersionSource(
            org_id=request.org_id,
            document_id=DocumentId(version.document_id),
            document_version_id=version.id,
            storage_object_id=stored_object.id,
            version_no=version.version_no,
            bucket=stored_object.bucket,
            object_key=stored_object.object_key,
            filename=version.filename,
            mime_type=version.mime_type,
            processing_status=processing_status,
            size_bytes=stored_object.size_bytes,
            checksum_sha256=stored_object.checksum_sha256,
            valid_from=version.valid_from,
            valid_to=version.superseded_at,
        )

    async def _mark_indexed(
        self,
        request: IngestDocumentRequest,
    ) -> None:
        async with self._doc_uow.begin() as transaction:
            await transaction.document_versions.set_processing_status(
                org_id=request.org_id,
                document_version_id=request.document_version_id,
                status=DocumentProcessingStatus.INDEXED,
            )

    async def _mark_failed(
        self,
        request: IngestDocumentRequest,
    ) -> None:
        try:
            async with self._doc_uow.begin() as transaction:
                await transaction.document_versions.set_processing_status(
                    org_id=request.org_id,
                    document_version_id=(request.document_version_id),
                    status=DocumentProcessingStatus.FAILED,
                )
        except Exception:
            logger.exception(
                "Could not persist failed ingestion status",
                extra={
                    "document_version_id": (request.document_version_id),
                },
            )
