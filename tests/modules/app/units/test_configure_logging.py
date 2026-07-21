from __future__ import annotations

import logging

from src.shared.configs.logger import configure_logging
from src.shared.configs.settings import LoggingSettings


def test_configure_logging_raises_httpx_and_httpcore_above_info() -> None:
    configure_logging(
        LoggingSettings(handlers=["console"], root_level="INFO"),
        service_name="evaluation-run",
    )

    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING
