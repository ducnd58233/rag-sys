from __future__ import annotations

import logging
from enum import Enum
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.shared.http.middlewares import REQUEST_ID_HEADER

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    INTERNAL = "INTERNAL"
    NOT_FOUND = "NOT_FOUND"
    VALIDATION = "VALIDATION"
    CONFLICT = "CONFLICT"


class AppError(Exception):
    code: str = ErrorCode.INTERNAL.value
    status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    default_message: str = "Internal error"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        code: str | None = None,
        status: int | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details or {}
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class InternalError(AppError):
    code = ErrorCode.INTERNAL.value
    status = HTTPStatus.INTERNAL_SERVER_ERROR
    default_message = "Internal error"


class NotFoundError(AppError):
    code = ErrorCode.NOT_FOUND.value
    status = HTTPStatus.NOT_FOUND
    default_message = "Resource not found"


class ValidationError(AppError):
    code = ErrorCode.VALIDATION.value
    status = HTTPStatus.UNPROCESSABLE_ENTITY
    default_message = "Validation failed"


class ConflictError(AppError):
    code = ErrorCode.CONFLICT.value
    status = HTTPStatus.CONFLICT
    default_message = "Conflict"


def build_error_response(request: Request, err: AppError) -> JSONResponse:
    body = err.to_dict()
    request_id = request.headers.get(REQUEST_ID_HEADER)
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    if request_id:
        body["request_id"] = request_id
    return JSONResponse(status_code=err.status, content=body, headers=headers)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return build_error_response(request, exc)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", exc_info=exc)
    return build_error_response(request, InternalError())


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)