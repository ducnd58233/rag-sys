import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from uvicorn.logging import DefaultFormatter

from src.shared.configs.settings import LoggingSettings
from src.shared.observability.logging import TraceContextJsonFormatter


_NOISY_HTTP_LOGGERS = ("httpx", "httpcore")


def configure_logging(config: LoggingSettings, *, service_name: str):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(config.root_level or "INFO")
    file_formatter: logging.Formatter
    if config.formatter == "json":
        file_formatter = TraceContextJsonFormatter(service_name=service_name)
    else:
        file_formatter = logging.Formatter(
            config.text_format or logging.BASIC_FORMAT,
            config.datefmt,
        )
    console_formatter = DefaultFormatter(
        fmt="%(levelprefix)s %(message)s",
        use_colors=None,
    )
    for handler in config.handlers or []:
        if handler == "console":
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            root.addHandler(console_handler)
        elif handler == "file":
            file_path = Path(config.file_path).with_stem(service_name)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=config.file_max_bytes,
                backupCount=config.file_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(file_formatter)
            root.addHandler(file_handler)

    # httpx logs every successful request at INFO ("HTTP Request: POST ... 200 OK"),
    # which drowns eval-run / eval-prepare progress (judge + graph extraction).
    for name in _NOISY_HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
