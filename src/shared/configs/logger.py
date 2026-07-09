import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from configs.settings import LoggingSettings


def configure_logging(config: LoggingSettings):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(config.root_level or "INFO")
    formatter = logging.Formatter(
        config.format or logging.BASIC_FORMAT,
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
