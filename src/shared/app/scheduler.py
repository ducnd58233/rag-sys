from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

PeriodicTask = Callable[[], Awaitable[None]]


class PeriodicRuntime:
    def __init__(
        self,
        *,
        task: PeriodicTask,
        interval_seconds: float,
        name: str,
    ) -> None:
        self._task = task
        self._interval_seconds = interval_seconds
        self._name = name
        self._stopping = asyncio.Event()

    async def run(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._task()
            except Exception:
                logger.exception("Periodic task '%s' failed", self._name)
            await self._wait_for_next_run_or_stop()

    def request_stop(self) -> None:
        self._stopping.set()

    async def _wait_for_next_run_or_stop(self) -> None:
        try:
            await asyncio.wait_for(
                self._stopping.wait(),
                timeout=self._interval_seconds,
            )
        except TimeoutError:
            pass
