import asyncio
import logging

from src.modules.ingestion.domain.errors import IngestionInternalError, IngestionValidationError
from src.modules.ingestion.domain.models import Chunk, ChunkDraft, DocumentSource, ProcessedDocument
from src.modules.ingestion.infra.processors.unstructured.config import UnstructuredProcessorRuntimeConfig
from src.modules.ingestion.infra.processors.unstructured.metadata_extractor import extract_element_metadata


logger = logging.getLogger(__name__)

CHUNKING_STRATEGY_NAME = "by_title"

class UnstructuredDocumentProcessor:
    def __init__(self, config: UnstructuredProcessorRuntimeConfig) -> None:
        self._config = config
    
    async def process(self, source: DocumentSource) -> ProcessedDocument:
        from unstructured.chunking.title import chunk_by_title
        try:
            raw_elements = await asyncio.wait_for(
                asyncio.to_thread(self._partition, source),
                timeout=self._config.extraction_timeout_seconds,
            )
            chunked_elements = await asyncio.to_thread(
                self._chunk_elements,
                raw_elements,
            )
        except TimeoutError as e:
            raise IngestionInternalError(
                message=(
                    "Document extraction timed out after "
                    f"{self._config.extraction_timeout_seconds}s: {source.source_uri}",
                ),
            ) from e
        except Exception as e:
            raise IngestionInternalError(
                message=f"Failed to process document {source.source_uri}: {e}",
            ) from e

        base_metadata = {
            "source_uri": source.source_uri,
            "filename": source.filename,
            "mime_type": source.mime_type,
            "chunking_strategy": CHUNKING_STRATEGY_NAME,
            "processor": "unstructured",
        }
        drafts = tuple(
            ChunkDraft(
                content=element.text,
                metadata=extract_element_metadata(element),
            )
            for element in chunked_elements
        )

        chunks: list[Chunk] = []
        for i, draft in enumerate(drafts):
            chunk = Chunk.from_draft(
                document_id=source.document_id,
                version_no=source.version_no,
                index=i,
                base_metadata=base_metadata,
                draft=draft,
                chunking_strategy=CHUNKING_STRATEGY_NAME,
            )
            if chunk is not None:
                chunks.append(chunk)

        logger.info(
            "Processed document_id=%s elements=%d chunks=%d",
            source.document_id.value,
            len(raw_elements),
            len(chunks),
        )
        return ProcessedDocument(
            document_id=source.document_id,
            metadata=base_metadata,
            chunks=tuple(chunks),
        )

    def _chunk_elements(self, elements: list[object]) -> list[Chunk]:
        from unstructured.chunking.title import chunk_by_title

        return chunk_by_title(
            elements=elements,
            max_characters=self._config.chunking.max_characters,
            combine_text_under_n_chars=self._config.chunking.combine_text_under_n_chars,
            new_after_n_chars=self._config.chunking.new_after_n_chars,
        )

    def _partition(self, source: DocumentSource) -> list[object]:
        from unstructured.partition.auto import partition
        from io import BytesIO
        
        strategy = self._config.partition.pdf_strategy

        if source.local_path is not None:
            return partition(filename=str(source.local_path), strategy=strategy)
        
        if source.content is not None:
            return partition(
                file=BytesIO(source.content),
                metadata_filename=source.filename,
                strategy=strategy,
            )
        raise IngestionValidationError(
            message="DocumentSource must provide local_path or content",
        )