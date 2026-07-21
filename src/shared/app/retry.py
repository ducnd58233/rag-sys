from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

from src.shared.kernel.errors import DomainException, ErrorCode

logger = logging.getLogger(__name__)

T = TypeVar("T")

NON_RETRYABLE_CODES = frozenset(
    {ErrorCode.VALIDATION, ErrorCode.NOT_FOUND, ErrorCode.CONFLICT},
)

_TRANSIENT_HTTPX = (
    httpx.TransportError,
    httpx.TimeoutException,
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


def is_transient_io_error(error: Exception) -> bool:
    if isinstance(error, _TRANSIENT_HTTPX):
        return True
    cause = error.__cause__ or error.__context__
    if isinstance(cause, Exception) and cause is not error:
        return is_transient_io_error(cause)
    return False


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[Exception], bool],
    operation_name: str = "operation",
) -> T:
    attempt = 1
    while True:
        try:
            return await operation()
        except Exception as error:
            if not is_retryable(error) or attempt >= policy.max_attempts:
                raise
            delay = policy.delay_for(attempt)
            logger.warning(
                "%s failed (%s); retry %s/%s in %.2fs",
                operation_name,
                error.__class__.__name__,
                attempt,
                policy.max_attempts,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1
