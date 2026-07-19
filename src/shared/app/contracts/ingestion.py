from __future__ import annotations

import json
from dataclasses import dataclass


class InvalidIngestionEventError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class IngestionRequestedEvent:
    org_id: int
    document_id: int
    document_version_id: int
    version_no: int

    def encode(self) -> bytes:
        return json.dumps(
            {
                "org_id": self.org_id,
                "document_id": self.document_id,
                "document_version_id": self.document_version_id,
                "version_no": self.version_no,
            },
        ).encode("utf-8")

    @classmethod
    def decode(cls, payload: bytes) -> IngestionRequestedEvent:
        try:
            data = json.loads(payload)
            return cls(
                org_id=int(data["org_id"]),
                document_id=int(data["document_id"]),
                document_version_id=int(data["document_version_id"]),
                version_no=int(data["version_no"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise InvalidIngestionEventError(
                "Malformed ingestion requested event payload",
            ) from error
