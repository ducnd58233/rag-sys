from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

class ErrorCode(StrEnum):
    INTERNAL = "INTERNAL"
    NOT_FOUND = "NOT_FOUND"
    VALIDATION = "VALIDATION"
    CONFLICT = "CONFLICT"

class DomainException(Exception):
    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.message = message
        self.details = dict(details or {})
        super().__init__(message)

class ValidationDomainError(DomainException):
    code = ErrorCode.VALIDATION

class NotFoundDomainError(DomainException):
    code = ErrorCode.NOT_FOUND

class ConflictDomainError(DomainException):
    code = ErrorCode.CONFLICT

class InternalDomainError(DomainException):
    code = ErrorCode.INTERNAL