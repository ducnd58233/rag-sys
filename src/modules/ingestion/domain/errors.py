from src.shared.kernel.errors import (
    ConflictDomainError,
    DomainException,
    InternalDomainError,
    NotFoundDomainError,
    ValidationDomainError,
)

class IngestionError(DomainException):
    pass

class IngestionValidationError(IngestionError, ValidationDomainError):
    pass

class IngestionNotFoundError(IngestionError, NotFoundDomainError):
    pass

class IngestionConflictError(IngestionError, ConflictDomainError):
    pass

class IngestionInternalError(IngestionError, InternalDomainError):
    pass