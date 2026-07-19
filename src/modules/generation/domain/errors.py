from src.shared.kernel.errors import (
    DomainException,
    NotFoundDomainError,
    ValidationDomainError,
)


class GenerationError(DomainException):
    pass


class GenerationValidationError(GenerationError, ValidationDomainError):
    pass


class GenerationNotFoundError(GenerationError, NotFoundDomainError):
    pass
