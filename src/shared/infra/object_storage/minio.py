import asyncio
from datetime import timedelta
from pathlib import Path
from minio import Minio, S3Error

from src.shared.app.ports import StoredObject
from src.shared.app.ports.object_storage import StorageBucket
from src.shared.configs.settings import ObjectStorageSettings


class ObjectStorageError(RuntimeError):
    pass

class MinioObjectStorage:
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
        ttl_seconds: int,
    ) -> str:
        try:
            return await asyncio.to_thread(
                self._client.presigned_put_object,
                bucket.value,
                key,
                timedelta(seconds=ttl_seconds),
            )
        except S3Error as e:
            raise ObjectStorageError(
                f"Could not create upload URL for {key}"
            ) from e

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
        except S3Error as e:
            raise ObjectStorageError(
                f"Could not create download URL for {key}"
            ) from e

    async def stat(self, bucket: StorageBucket, key: str) -> StoredObject:
        try:
            item = await asyncio.to_thread(
                self._client.stat_object,
                bucket.value,
                key,
            )
            return StoredObject(
                key=key,
                size_bytes=item.size,
                etag=item.etag,
                content_type=item.content_type,
                last_modified=item.last_modified,
            )
        except S3Error as e:
            raise ObjectStorageError(
                f"Could not stat object {key}"
            ) from e

    async def download_to_path(self, bucket: StorageBucket, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            await asyncio.to_thread(
                self._client.fget_object,
                bucket.value,
                key,
                str(destination),
            )
        except S3Error as e:
            raise ObjectStorageError(
                f"Could not download object {key} to {destination}"
            ) from e

    async def delete(self, bucket: StorageBucket, key: str) -> None:
        try:
            await asyncio.to_thread(
                self._client.remove_object,
                bucket.value,
                key,
            )
        except S3Error as e:
            raise ObjectStorageError(
                f"Could not delete object {key}"
            ) from e
            