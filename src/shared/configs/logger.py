import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.shared.configs.settings import LoggingSettings
from src.shared.observability.logging import TraceContextJsonFormatter


def configure_logging(config: LoggingSettings, *, service_name: str):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(config.root_level or "INFO")
    formatter: logging.Formatter
    if config.formatter == "json":
        formatter = TraceContextJsonFormatter(service_name=service_name)
    else:
        formatter = logging.Formatter(
            config.text_format or logging.BASIC_FORMAT,
            config.datefmt,
        )
    for handler in config.handlers or []:
        if handler == "console":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root.addHandler(console_handler)
        elif handler == "file":
            file_path = Path(config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=config.file_max_bytes,
                backupCount=config.file_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
