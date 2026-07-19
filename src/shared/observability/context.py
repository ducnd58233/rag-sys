from __future__ import annotations

from contextvars import ContextVar

from opentelemetry import trace

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("request_id", request_id)


def get_request_id() -> str | None:
    return _request_id.get()
