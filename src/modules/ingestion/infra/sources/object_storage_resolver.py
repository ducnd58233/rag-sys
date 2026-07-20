from __future__ import annotations

import asyncio
import hashlib
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from src.modules.ingestion.domain.errors import (
    IngestionInternalError,
    IngestionValidationError,
)
from src.modules.ingestion.domain.models import DocumentSource, DocumentVersionSource
from src.shared.app.ports import IObjectStorage
from src.shared.app.ports.object_storage import StorageBucket
from src.shared.infra.object_storage import ObjectStorageError


class ObjectStorageSourceResolver:
    def __init__(
        self,
        storage: IObjectStorage,
        max_file_size_bytes: int,
    ) -> None:
        self._storage = storage
        self._max_file_size_bytes = max_file_size_bytes

    @asynccontextmanager
    async def open(
        self,
        source: DocumentVersionSource,
    ) -> AsyncIterator[DocumentSource]:
        try:
            bucket = StorageBucket(source.bucket)
        except ValueError as error:
            raise IngestionValidationError(
                message=f"Unsupported storage bucket: {source.bucket}",
            ) from error

        filename = Path(source.filename).name
        if not filename:
            raise IngestionValidationError(
                message="Document filename cannot be empty",
            )

        if source.size_bytes > self._max_file_size_bytes:
            raise IngestionValidationError(
                message="Document exceeds maximum ingestion size",
            )

        try:
            stored = await self._storage.stat(
                bucket,
                source.object_key,
            )

            if stored.size_bytes != source.size_bytes:
                raise IngestionValidationError(
                    message="Stored object size does not match database metadata",
                )

            with tempfile.TemporaryDirectory(
                prefix="rag-ingestion-",
            ) as directory:
                local_path = Path(directory) / filename

                await self._storage.download_to_path(
                    bucket=bucket,
                    key=source.object_key,
                    destination=local_path,
                )

                if source.checksum_sha256 is not None:
                    checksum = await self._sha256(local_path)
                    if checksum != source.checksum_sha256.lower():
                        raise IngestionValidationError(
                            message="Stored object checksum does not match",
                        )

                yield DocumentSource(
                    document_id=source.document_id,
                    version_no=source.version_no,
                    filename=filename,
                    mime_type=source.mime_type,
                    source_uri=(f"s3://{bucket.value}/{source.object_key}"),
                    local_path=local_path,
                )

        except ObjectStorageError as error:
            raise IngestionInternalError(
                message="Could not read document from object storage",
            ) from error

    async def _sha256(self, path: Path) -> str:
        return await asyncio.to_thread(self._sha256_sync, path)

    def _sha256_sync(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
