from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

from minio import Minio

from src.shared.app.ports.object_storage import (
    ObjectStorageError,
    StorageBucket,
    StoredObject,
)
from src.shared.configs.settings import ObjectStorageSettings


class MinioObjectStorage:
    """MinIO implementation of the object-storage port."""

    def __init__(self, settings: ObjectStorageSettings) -> None:
        self._client = Minio(
            settings.endpoint,
            access_key=settings.access_key.get_secret_value(),
            secret_key=settings.secret_key.get_secret_value(),
            secure=settings.secure,
            region=settings.region,
        )

    async def presign_put(
        self,
        bucket: StorageBucket,
        key: str,
        *,
        ttl_seconds: int,
    ) -> str:
        try:
            return await asyncio.to_thread(
                self._client.presigned_put_object,
                bucket.value,
                key,
                timedelta(seconds=ttl_seconds),
            )
        except Exception as error:
            raise ObjectStorageError(
                f"Could not create upload URL for {key}",
            ) from error

    async def presign_get(
        self,
        bucket: StorageBucket,
        key: str,
        *,
        ttl_seconds: int,
    ) -> str:
        try:
            return await asyncio.to_thread(
                self._client.presigned_get_object,
                bucket.value,
                key,
                timedelta(seconds=ttl_seconds),
            )
        except Exception as error:
            raise ObjectStorageError(
                f"Could not create download URL for {key}",
            ) from error

    async def stat(
        self,
        bucket: StorageBucket,
        key: str,
    ) -> StoredObject:
        try:
            item = await asyncio.to_thread(
                self._client.stat_object,
                bucket.value,
                key,
            )
        except Exception as error:
            raise ObjectStorageError(
                f"Could not stat object {key}",
            ) from error

        if item.size is None or item.etag is None or item.last_modified is None:
            raise ObjectStorageError(
                f"Object metadata is incomplete for {key}",
            )

        return StoredObject(
            key=key,
            size_bytes=item.size,
            etag=item.etag,
            content_type=item.content_type,
            last_modified=item.last_modified,
        )

    async def download_to_path(
        self,
        bucket: StorageBucket,
        key: str,
        destination: Path,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            await asyncio.to_thread(
                self._client.fget_object,
                bucket.value,
                key,
                str(destination),
            )
        except Exception as error:
            raise ObjectStorageError(
                f"Could not download object {key}",
            ) from error

    async def delete(
        self,
        bucket: StorageBucket,
        key: str,
    ) -> None:
        try:
            await asyncio.to_thread(
                self._client.remove_object,
                bucket.value,
                key,
            )
        except Exception as error:
            raise ObjectStorageError(
                f"Could not delete object {key}",
            ) from error
