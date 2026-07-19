from src.shared.kernel.errors import (
    ConflictDomainError,
    DomainException,
    InternalDomainError,
    NotFoundDomainError,
    ValidationDomainError,
)


class DocumentError(DomainException):
    pass


class DocumentValidationError(DocumentError, ValidationDomainError):
    pass


class DocumentNotFoundError(DocumentError, NotFoundDomainError):
    pass


class DocumentConflictError(DocumentError, ConflictDomainError):
    pass


class DocumentInternalError(DocumentError, InternalDomainError):
    pass
