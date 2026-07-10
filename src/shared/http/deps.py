from __future__ import annotations

from fastapi import Request
from src.bootstrap import AppContainer


def get_container(request: Request) -> AppContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("App container is not initialized")
    return container