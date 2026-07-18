import asyncio
import logging
from src.modules.document.app.ports import IDocumentUnitOfWork
from src.modules.document.domain.models import DocumentProcessingStatus, DocumentScanStatus, StoredObjectStatus
from src.modules.ingestion.app.dto import IngestDocumentRequest, IngestDocumentResult
from src.modules.ingestion.app.ports import IDocumentProcessor, ISourceResolver, IVectorStore
from src.modules.ingestion.domain.errors import IngestionConflictError, IngestionInternalError, IngestionNotFoundError
from src.modules.ingestion.domain.models import DocumentId, DocumentVersionSource, IngestionStatus
from src.shared.app.ports import IEmbeddingModel


logger = logging.getLogger(__name__)

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

    async def execute(self, request: IngestDocumentRequest) -> IngestDocumentResult:
        source = await self._load_source_and_mark_parsing(request)

        try:
            async with self._source_resolver.open(source) as document_source:
                processed = await self._processor.process(document_source)

            chunks = processed.chunks
            vectors = (
                await self._embedder.embed(
                    [chunk.content for chunk in chunks],
                )
                if chunks
                else []
            )

            if len(vectors) != len(chunks):
                raise IngestionInternalError(
                    message='Embedding count does not match chunk count',
                )

            metadata = {
                **processed.metadata,
                'org_id': str(source.org_id),
                'case_id': str(source.case_id),
                'document_id': str(source.document_id.value),
                'document_version_id': str(source.document_version_id),
                'version': str(source.version_no),
                'filename': source.filename,
                'mime_type': source.mime_type,
            }

            await self._vector_store.create_index_if_not_exists()
            await self._vector_store.delete_by_document_id(
                source.document_id,
            )

            if chunks:
                await self._vector_store.upsert(
                    document_id=source.document_id,
                    metadata=metadata,
                    chunks=chunks,
                    vectors=vectors,
                )
        except Exception as e:
            await self._mark_failed(request)
            raise

        await self._mark_indexed(request)

        return IngestDocumentResult(
            document_id=source.document_id.value,
            document_version_id=source.document_version_id,
            version_no=source.version_no,
            chunk_count=len(chunks),
            status=IngestionStatus.COMPLETED,
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
            
            document, stored_object = await asyncio.gather(
                transaction.documents.get(
                    org_id=request.org_id,
                    document_id=version.document_id,
                ),
                transaction.stored_objects.get(
                    org_id=request.org_id,
                    stored_object_id=version.storage_object_id,
                ),
            )

            if document is None or stored_object is None:
                raise IngestionNotFoundError(
                    message="Document source is incomplete"
                )
            
            if version.scan_status is not DocumentScanStatus.CLEAN:
                raise IngestionConflictError(
                    message="Document version has not passed malware scan"
                )
            
            if stored_object.status is not StoredObjectStatus.AVAILABLE:
                raise IngestionConflictError(
                    message="Stored object is not available"
                )
            
            await transaction.document_versions.set_processing_status(
                org_id=request.org_id,
                document_version_id=request.document_version_id,
                status=DocumentProcessingStatus.PARSING,
            )

            return DocumentVersionSource(
                org_id=request.org_id,
                case_id=document.case_id,
                document_id=DocumentId(version.document_id),
                document_version_id=version.id,
                storage_object_id=stored_object.id,
                version_no=version.version_no,
                bucket=stored_object.bucket,
                object_key=stored_object.object_key,
                filename=version.filename,
                mime_type=version.mime_type,
                processing_status=DocumentProcessingStatus.PARSING,
                scan_status=version.scan_status,
                size_bytes=stored_object.size_bytes,
                checksum_sha256=stored_object.checksum_sha256,
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
                    document_version_id=request.document_version_id,
                    status=DocumentProcessingStatus.FAILED,
                )
        except Exception:
            logger.exception(
                'Could not persist failed ingestion status',
                extra={
                    'document_version_id': request.document_version_id,
                },
            )