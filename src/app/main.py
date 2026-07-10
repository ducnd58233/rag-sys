from collections.abc import Iterator
import importlib
import pkgutil
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from src.bootstrap import build_container
from src.shared.http.errors import register_exception_handlers
from src.shared.http.middlewares import HttpMiddleware, request_context_middleware

HTTP_MIDDLEWARES: tuple[HttpMiddleware, ...] = (request_context_middleware,)
router = APIRouter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = build_container()
    try:
        yield
    finally:
        await app.state.container.shutdown()

def discover_module_routers() -> Iterator[APIRouter]:
    try:
        import src.modules
    except ModuleNotFoundError:
        return

    for module_info in pkgutil.walk_packages(
        src.modules.__path__,
        prefix=f"{src.modules.__name__}.",
    ):
        if not module_info.name.endswith(".api.router"):
            continue
        mod = importlib.import_module(module_info.name)
        router = getattr(mod, "router", None)
        if isinstance(router, APIRouter):
            yield router


def register_middlewares(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for middleware in reversed(HTTP_MIDDLEWARES):
        app.middleware("http")(middleware)


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}

def create_app() -> FastAPI:
    app = FastAPI(
        title="Trading Bot",
        lifespan=lifespan,
    )

    register_middlewares(app)
    register_exception_handlers(app)
    api_v1 = APIRouter(prefix="/api/v1")

    api_v1.include_router(router)

    for module_router in discover_module_routers():
        api_v1.include_router(module_router)

    app.include_router(api_v1)
    return app


def run() -> None:
    uvicorn.run(
        "src.app.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
    )


def run_dev() -> None:
    uvicorn.run(
        "src.app.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
