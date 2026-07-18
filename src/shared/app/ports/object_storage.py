from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

class PresignTtlSeconds:
    UPLOAD = 900
    DOWNLOAD = 300
    EXPORT = 600

class StorageBucket(StrEnum):
    DOCUMENTS = "documents"
    EXPORTS = "exports"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class StoredObject:
    key: str
    size_bytes: int
    etag: str
    content_type: str | None
    last_modified: datetime

class IObjectStorage(Protocol):
    async def presign_put(
        self,
        bucket: StorageBucket,
        key: str,
        *,
        ttl_seconds: int,
    ) -> str: ...

    async def presign_get(
        self,
        bucket: StorageBucket,
        key: str,
        *,
        ttl_seconds: int,
    ) -> str: ...

    async def stat(self, bucket: StorageBucket, key: str) -> StoredObject: ...

    async def download_to_path(
        self, bucket: StorageBucket, key: str, destination: Path,
    ) -> None: ...

    async def delete(self, bucket: StorageBucket, key: str) -> None: ...