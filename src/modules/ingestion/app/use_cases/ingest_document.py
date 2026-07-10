from datetime import datetime, timezone
import logging
import uuid
from src.modules.ingestion.app.dto import IngestDocumentRequest, IngestDocumentResult
from src.modules.ingestion.app.ports import ISourceResolver, IDocumentExtractor, IChunkingStrategy, IVectorStore
from src.modules.ingestion.domain.models import DocumentId, IngestionStatus
from src.shared.infra.embedding import IEmbeddingModel


logger = logging.getLogger(__name__)

class IngestDocumentUseCase:
    def __init__(
        self,
        source_resolver: ISourceResolver,
        extractor: IDocumentExtractor,
        chunking_strategy: IChunkingStrategy,
        embedder: IEmbeddingModel,
        vector_store: IVectorStore,
    ) -> None:
        self._source_resolver = source_resolver
        self._extractor = extractor
        self._chunking_strategy = chunking_strategy
        self._embedder = embedder
        self._vector_store = vector_store

    async def execute(self, request: IngestDocumentRequest) -> IngestDocumentResult:
        document_id = DocumentId(
            value=request.document_id or str(uuid.uuid4()),
        )

        logger.info("Starting ingestion for document_id=%s", document_id.value)

        source = await self._source_resolver.resolve(
            source_path=request.source_path,
            document_id=document_id,
        )

        extraction = await self._extractor.extract(source)
        merged_metadata = {
            **extraction.metadata,
            **request.metadata,
        }

        chunks = self._chunking_strategy.chunk(extraction)
        if not chunks:
            logger.warning("No chunks generated for document_id=%s", document_id.value)
            return IngestDocumentResult(
                document_id=document_id.value,
                chunk_count=0,
                status=IngestionStatus.COMPLETED,
            )

        vectors = await self._embedder.embed([chunk.content for chunk in chunks])
        await self._vector_store.create_index_if_not_exists()
        await self._vector_store.delete_by_document_id(document_id)
        await self._vector_store.upsert(
            document_id=document_id,
            metadata=merged_metadata,
            chunks=chunks,
            vectors=vectors,
        )

        logger.info(
            "Completed ingestion document_id=%s chunk_count=%d indexed_at=%s",
            document_id.value,
            len(chunks),
            datetime.now(timezone.utc).isoformat(),
        )
        return IngestDocumentResult(
            document_id=document_id.value,
            chunk_count=len(chunks),
            status=IngestionStatus.COMPLETED,
        )