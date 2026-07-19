from __future__ import annotations

from src.shared.kernel.errors import (
    DomainException,
    InternalDomainError,
    ValidationDomainError,
)


class RetrievalError(DomainException):
    pass


class RetrievalValidationError(RetrievalError, ValidationDomainError):
    pass


class RetrievalInternalError(RetrievalError, InternalDomainError):
    pass
