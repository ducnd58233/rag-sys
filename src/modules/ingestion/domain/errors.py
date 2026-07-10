from src.shared.http.errors import ConflictError, InternalError, NotFoundError, ValidationError


class IngestionValidationError(ValidationError):
    default_message = "Ingestion validation failed"

class IngestionNotFoundError(NotFoundError):
    default_message = "Document source not found"

class IngestionConflictError(ConflictError):
    default_message = "Ingestion conflict"

class IngestionInternalError(InternalError):
    default_message = "Ingestion failed"