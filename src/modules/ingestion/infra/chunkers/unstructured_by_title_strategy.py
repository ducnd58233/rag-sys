import logging
from unstructured.chunking.title import chunk_by_title

from src.modules.ingestion.domain.errors import IngestionInternalError
from src.modules.ingestion.domain.models import Chunk, ExtractionResult
from src.modules.ingestion.infra.configs import ByTitleChunkingConfig
from src.modules.ingestion.infra.mapper.unstructured_element_mapper import to_unstructured
from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)

class UnstructuredByTitleStrategy:
    def __init__(self, config: ByTitleChunkingConfig) -> None:
        self._config = config

    def chunk(self, extraction: ExtractionResult) -> list[Chunk]:
        if not extraction.elements:
            return []

        try:
            unstructured_elements = [
                to_unstructured(element) for element in extraction.elements
            ]
            chunked_elements = chunk_by_title(
                elements=unstructured_elements,
                max_characters=self._config.max_characters,
                combine_text_under_n_chars=self._config.combine_text_under_n_chars,
                new_after_n_chars=self._config.new_after_n_chars,
            )
        except Exception as e:
            raise IngestionInternalError(f"Failed to chunk elements: {e}") from e

        chunks: list[Chunk] = []
        for idx, ele in enumerate(chunked_elements):
            text = str(ele).strip()
            if not text:
                continue
            
            element_metadata = self._element_metadata(ele)
            chunks.append(
                Chunk(
                    chunk_id=f"{extraction.document_id.value}:{idx}",
                    document_id=extraction.document_id,
                    index=idx,
                    content=text,
                    metadata={
                        **extraction.metadata,
                        **element_metadata,
                        "chunk_index": str(idx),
                        "chunking_strategy": "by_title",
                    },
                ),
            )
        logger.info(
            "Chunked document_id=%s chunks=%d",
            extraction.document_id.value,
            len(chunks),
        )
        return chunks

    def _element_metadata(self, element: Element) -> dict[str, str]:
        metadata: dict[str, str] = {}
        if hasattr(element, "category"):
            metadata["element_category"] = str(element.category)
        raw_metadata = getattr(element, "metadata", None)
        if raw_metadata is not None and hasattr(raw_metadata, "to_dict"):
            for key, value in raw_metadata.to_dict().items():
                if value is not None:
                    metadata[str(key)] = str(value)
        return metadata