"""Structured logging helpers.

Every log record carries structured fields (run_id, dataset, subject, ...)
through ``extra``. Sensor arrays are never logged.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

Logger = logging.Logger

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> Logger:
    """Return a logger; logging is auto-configured."""
    configure_logging()
    return logging.getLogger(name)


def log_structured(
    logger: Logger,
    level: int,
    message: str,
    /,
    **fields: Any,
) -> None:
    """Log ``message`` with structured JSON fields stored in ``extra['fields']``."""

    clean: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = str(value)
    logger.log(
        level,
        "%s | %s",
        message,
        json.dumps(clean, ensure_ascii=False, default=str) if clean else "{}",
        extra={"fields": clean},
    )
