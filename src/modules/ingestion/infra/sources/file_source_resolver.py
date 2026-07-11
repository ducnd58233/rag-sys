from __future__ import annotations

import mimetypes
from pathlib import Path

from src.modules.ingestion.domain.errors import IngestionNotFoundError, IngestionValidationError
from src.modules.ingestion.domain.models import DocumentId, DocumentSource
from src.modules.ingestion.infra.configs import SourceResolverConfig

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".md", ".pdf", ".docx", ".html", ".htm"},
)


class FileSourceResolver:
    def __init__(self, config: SourceResolverConfig) -> None:
        self._config = config

    async def resolve(
        self,
        source_path: str,
        document_id: DocumentId | None = None,
    ) -> DocumentSource:
        path = Path(source_path).expanduser().resolve()

        if not path.is_file():
            raise IngestionNotFoundError(
                message=f"File not found: {path}",
                details={"file_path": str(path)},
            )

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise IngestionValidationError(
                message=f"Unsupported file extension: {suffix}",
                details={
                    "file_extension": suffix,
                    "supported_extensions": ",".join(sorted(SUPPORTED_EXTENSIONS)),
                },
            )

        size = path.stat().st_size
        if size > self._config.max_file_size_bytes:
            raise IngestionValidationError(
                message=f"File exceeds max size {self._config.max_file_size_bytes} bytes: {path}",
                details={
                    "max_file_size_bytes": str(self._config.max_file_size_bytes),
                    "file_size_bytes": str(size),
                    "file_path": str(path),
                },
            )

        mime_type = self._resolve_mime_type(path)
        resolved_id = document_id or DocumentId(value=path.stem)

        return DocumentSource(
            document_id=resolved_id,
            filename=path.name,
            mime_type=mime_type,
            source_uri=str(path),
            local_path=path,
            content=None,
        )

    def _resolve_mime_type(self, path: Path) -> str:
        if path.suffix.lower() == ".md":
            return "text/markdown"
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"