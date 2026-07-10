import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

logger = logging.getLogger(__name__)

CallNext = Callable[[Request], Awaitable[Response]]

HttpMiddleware = Callable[
    [Request, CallNext],
    Awaitable[Response],
]

REQUEST_ID_HEADER = "X-Request-Id"


async def request_context_middleware(request: Request, call_next: CallNext) -> Response:
    request_id = request.headers.get(REQUEST_ID_HEADER)
    start = time.perf_counter()
    if not request_id:
        request_id = str(uuid.uuid4())
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    response.headers[REQUEST_ID_HEADER] = request_id
    return response