from __future__ import annotations

from typing import Protocol


class IIdGenerator(Protocol):
    def next_id(self) -> int: ...
