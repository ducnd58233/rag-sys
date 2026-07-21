from __future__ import annotations

import httpx
import pytest

from src.shared.app.retry import (
    RetryPolicy,
    is_transient_io_error,
    run_with_retry,
)
from src.shared.kernel.errors import ValidationDomainError


def test_retry_policy_delay_caps_at_max() -> None:
    policy = RetryPolicy(
        max_attempts=5,
        base_delay_seconds=1.0,
        max_delay_seconds=4.0,
    )

    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(3) == 4.0
    assert policy.delay_for(4) == 4.0


def test_is_transient_io_error_for_httpx_read_error() -> None:
    assert is_transient_io_error(httpx.ReadError("closed"))
    assert is_transient_io_error(httpx.ConnectError("refused"))
    assert not is_transient_io_error(ValueError("bad input"))
    assert not is_transient_io_error(ValidationDomainError("invalid"))


@pytest.mark.asyncio
async def test_run_with_retry_retries_transient_errors_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("src.shared.app.retry.asyncio.sleep", fake_sleep)

    attempts = {"count": 0}

    async def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.ReadError("peer closed")
        return "ok"

    result = await run_with_retry(
        flaky,
        policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.5,
            max_delay_seconds=2.0,
        ),
        is_retryable=is_transient_io_error,
    )

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_run_with_retry_does_not_retry_non_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_delay: float) -> None:
        raise AssertionError("should not sleep")

    monkeypatch.setattr("src.shared.app.retry.asyncio.sleep", fake_sleep)

    async def boom() -> None:
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        await run_with_retry(
            boom,
            policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.1),
            is_retryable=is_transient_io_error,
        )
