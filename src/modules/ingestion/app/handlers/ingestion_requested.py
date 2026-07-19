from __future__ import annotations

from src.modules.ingestion.app.dto import IngestDocumentRequest
from src.modules.ingestion.app.use_cases.ingest_document import IngestDocumentUseCase
from src.shared.app.contracts.ingestion import IngestionRequestedEvent
from src.shared.app.ports.message_queue import IncomingMessage


class IngestionRequestedHandler:
    def __init__(self, ingest_document: IngestDocumentUseCase) -> None:
        self._ingest_document = ingest_document

    async def handle(self, message: IncomingMessage) -> None:
        event = IngestionRequestedEvent.decode(message.payload)
        await self._ingest_document.execute(
            IngestDocumentRequest(
                org_id=event.org_id,
                document_version_id=event.document_version_id,
            ),
        )
