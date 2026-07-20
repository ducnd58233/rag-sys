from __future__ import annotations

import functools
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TypeVar

from opentelemetry.metrics import Histogram
from opentelemetry.trace import Span, Status, StatusCode, Tracer

T = TypeVar("T")

_AttributeValue = str | bool | int | float
_Attributes = Mapping[str, _AttributeValue]


@asynccontextmanager
async def traced(
    tracer: Tracer,
    name: str,
    *,
    attributes: _Attributes | None = None,
    duration_metric: Histogram | None = None,
    metric_attributes: _Attributes | None = None,
) -> AsyncIterator[Span]:
    """Wrap a block in a span, recording duration and outcome into
    ``duration_metric`` on the way out. Generalizes the
    started_at/try/except/finally skeleton that used to be repeated at
    every span-creation site in this codebase.

    The yielded span is the real span, so callers still set
    result-derived attributes on it inline, same as before.
    """
    started_at = time.perf_counter()
    outcome = "success"
    # set_status_on_exception defaults to True, which would overwrite our
    # own set_status below with the full exception message. We only want
    # the exception class name in the span status, not its message, so
    # telemetry doesn't end up carrying whatever was in the exception
    # text (which can include user input or other request details).
    with tracer.start_as_current_span(
        name,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as error:
            outcome = "failure"
            span.set_status(Status(StatusCode.ERROR, error.__class__.__name__))
            raise
        finally:
            if duration_metric is not None:
                duration_metric.record(
                    time.perf_counter() - started_at,
                    {**(metric_attributes or {}), "outcome": outcome},
                )


def traced_span(
    tracer: Tracer,
    name: str,
    *,
    attributes: _Attributes | None = None,
    duration_metric: Histogram | None = None,
    metric_attributes: _Attributes | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator form of `traced`, for the common case where the whole
    function body is the traced unit. Built on `traced`, not a separate
    implementation, so both forms stay behaviorally identical."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> T:
            async with traced(
                tracer,
                name,
                attributes=attributes,
                duration_metric=duration_metric,
                metric_attributes=metric_attributes,
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
