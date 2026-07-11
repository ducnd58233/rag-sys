from src.modules.ingestion.app.ports import IDocumentProcessor
from src.modules.ingestion.domain.errors import IngestionValidationError
from src.modules.ingestion.infra.configs import ByTitleChunkingConfig
from src.modules.ingestion.infra.processors.unstructured.config import UnstructuredPartitionConfig, UnstructuredProcessorRuntimeConfig
from src.modules.ingestion.infra.processors.unstructured.processor import UnstructuredDocumentProcessor
from src.shared.configs.settings import IngestionSettings

SUPPORTED_PROCESSORS = ("unstructured",)

class DocumentProcessorFactory:
    @staticmethod
    def create(settings: IngestionSettings) -> IDocumentProcessor:
        chunking = ByTitleChunkingConfig(
            max_characters=settings.chunk_max_characters,
            combine_text_under_n_chars=settings.chunk_combine_text_under_n_chars,
            new_after_n_chars=settings.chunk_new_after_n_chars,
        )
        match settings.processor.lower():
            case "unstructured":
                return UnstructuredDocumentProcessor(
                    UnstructuredProcessorRuntimeConfig(
                        chunking=chunking,
                        partition=UnstructuredPartitionConfig(
                            pdf_strategy=settings.pdf_strategy,
                        ),
                        extraction_timeout_seconds=settings.extraction_timeout_seconds,
                    ),
                )
            case unknown:
                raise IngestionValidationError(
                    message=f"Unsupported ingestion processor: {unknown}",
                    details={
                        "supported_processors": ",".join(SUPPORTED_PROCESSORS),
                    },
                )