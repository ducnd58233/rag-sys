from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from src.shared.http.middlewares import REQUEST_ID_HEADER
from src.shared.kernel.errors import DomainException, ErrorCode

logger = logging.getLogger(__name__)

__all__ = [
    "AppError",
    "ConflictError",
    "ErrorCode",
    "InternalError",
    "NotFoundError",
    "ValidationError",
    "build_error_response",
    "domain_exception_to_app_error",
    "register_exception_handlers",
]

class ErrorResponse(BaseModel):
    message: str
    details: dict[str, str] = Field(default_factory=dict)
    request_id: str | None = None
    
class AppError(Exception):
    code: str = ErrorCode.INTERNAL.value
    status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    default_message: str = "Internal error"

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, str] | None = None,
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

    def to_error_response(self, request_id: str | None = None) -> ErrorResponse:
        return ErrorResponse(
            message=self.message,
            details=self.details,
            request_id=request_id,
        )


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


_CODE_TO_APP_ERROR: dict[ErrorCode, type[AppError]] = {
    ErrorCode.INTERNAL: InternalError,
    ErrorCode.NOT_FOUND: NotFoundError,
    ErrorCode.VALIDATION: ValidationError,
    ErrorCode.CONFLICT: ConflictError,
}


def domain_exception_to_app_error(exc: DomainException) -> AppError:
    app_error_cls = _CODE_TO_APP_ERROR.get(exc.code, InternalError)
    return app_error_cls(message=exc.message, details=exc.details)


def build_error_response(request: Request, err: AppError) -> JSONResponse:
    request_id = request.headers.get(REQUEST_ID_HEADER)
    body = err.to_error_response(request_id=request_id)
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    return JSONResponse(
        status_code=err.status,
        content=body.model_dump(exclude_none=True),
        headers=headers,
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return build_error_response(request, exc)


async def domain_exception_handler(
    request: Request,
    exc: DomainException,
) -> JSONResponse:
    return build_error_response(request, domain_exception_to_app_error(exc))


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", exc_info=exc)
    return build_error_response(request, InternalError())


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(DomainException, domain_exception_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)