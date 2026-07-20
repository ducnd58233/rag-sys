from __future__ import annotations

from types import ModuleType

from scripts.evaluation.sources import bioasq

# See datasets/README.md ("Adding another dataset") for the module contract.
SOURCES: dict[str, ModuleType] = {
    bioasq.NAME: bioasq,
}


def get_source(name: str) -> ModuleType:
    try:
        return SOURCES[name]
    except KeyError:
        available = ", ".join(sorted(SOURCES))
        raise ValueError(f"unknown dataset {name!r}; available: {available}") from None
