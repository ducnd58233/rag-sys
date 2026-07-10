import asyncio
from io import BytesIO
import logging


from src.modules.ingestion.domain.errors import IngestionInternalError, IngestionValidationError
from src.modules.ingestion.domain.models import DocumentSource, ExtractedElement, ExtractionResult


logger = logging.getLogger(__name__)

class UnstructuredExtractor:
    async def extract(self, source: DocumentSource) -> ExtractionResult:
        try:
            elements = await asyncio.to_thread(self._partition, source)
        except Exception as e:
            raise IngestionInternalError(f"Failed to extract elements from {source.source_uri}: {e}") from e
        
        mapped = tuple(self._map_element(element) for element in elements)
        metadata = {
            "filename": source.filename,
            "source_uri": source.source_uri,
            "mime_type": source.mime_type,
            "extractor": "unstructured",
        }

        logger.info(
            "Extracted document_id=%s elements=%d",
            source.document_id.value,
            len(mapped),
        )
        return ExtractionResult(
            document_id=source.document_id,
            elements=mapped,
            metadata=metadata,
        )
    
    def _partition(self, source: DocumentSource) -> list[dict]:
        from unstructured.partition.auto import partition
        
        if source.local_path is not None:
            return partition(filename=str(source.local_path))

        if source.content is not None:
            return partition(
                file=BytesIO(source.content),
                metadata_filename=source.filename,
            )

        raise IngestionValidationError("DocumentSource must provide local_path or content")
    
    def _map_element(self, element: object) -> ExtractedElement:
        category = getattr(element, "category", "Unknown")
        metadata: dict[str, str] = {}
        raw_metadata = getattr(element, "metadata", None)
        if raw_metadata is not None and hasattr(raw_metadata, "to_dict"):
            for key, value in raw_metadata.to_dict().items():
                if value is not None:
                    metadata[str(key)] = str(value)
        return ExtractedElement(
            text=str(element).strip(),
            category=str(category),
            metadata=metadata,
        )
