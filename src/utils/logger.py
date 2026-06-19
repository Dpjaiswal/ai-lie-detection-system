"""
Logger — AI-Powered Lie Detection System
Structured logging with Loguru, supports JSON and text formats.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure the application logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to write log file
        json_format: If True, emit structured JSON logs
        rotation: Log file rotation policy
        retention: Log file retention policy
    """
    _logger.remove()  # Remove default handler

    # ── Console handler ───────────────────────────────────────
    if json_format:
        fmt = (
            '{"time":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
            '"level":"{level}",'
            '"name":"{name}",'
            '"message":"{message}"}'
        )
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    _logger.add(
        sys.stdout,
        level=level,
        format=fmt,
        colorize=not json_format,
        enqueue=True,
    )

    # ── File handler (optional) ───────────────────────────────
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_file,
            level=level,
            format=fmt,
            rotation=rotation,
            retention=retention,
            compression="gz",
            enqueue=True,
        )


def get_logger(name: str = "lie_detection"):
    """Return a contextualized logger bound to the given name."""
    return _logger.bind(module=name)


# ── Initialize from environment ───────────────────────────────
_log_level = os.getenv("LOG_LEVEL", "INFO")
_log_format = os.getenv("LOG_FORMAT", "text")
_log_file = os.getenv("LOG_FILE", "logs/app.log")

setup_logger(
    level=_log_level,
    log_file=_log_file,
    json_format=(_log_format == "json"),
)

logger = get_logger()
