"""Logging configuration using Loguru."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
    file_rotation: str = "10 MB",
    file_retention: str = "7 days",
    compression: str = "zip",
    serialize: bool = True,
) -> None:
    """Configure Loguru logger with JSON serialization and file rotation."""
    logger.remove()

    # Console logging
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        serialize=False,
    )

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # File logging with JSON serialization
        logger.add(
            log_path / "mnemo_{time:YYYY-MM-DD}.log",
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=file_rotation,
            retention=file_retention,
            compression=compression,
            serialize=serialize,
            enqueue=True,
        )


def get_logger(name: str):
    """Get a logger instance for a module."""
    return logger.bind(module=name)
