from __future__ import annotations

from dataclasses import dataclass

from src.shared.kernel.errors import DomainException, ErrorCode

NON_RETRYABLE_CODES = frozenset(
    {ErrorCode.VALIDATION, ErrorCode.NOT_FOUND, ErrorCode.CONFLICT},
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0

    def delay_for(self, attempt: int) -> float:
        return min(
            self.base_delay_seconds * (2 ** (attempt - 1)),
            self.max_delay_seconds,
        )


def is_retryable(error: Exception) -> bool:
    if isinstance(error, DomainException):
        return error.code not in NON_RETRYABLE_CODES
    return True
