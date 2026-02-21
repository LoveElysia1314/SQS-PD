"""Logging helpers for sqs_pd.

Goal: keep library output quiet by default, and only emit
logs to console when the public API asks for it (e.g. console_log=True).

We intentionally avoid configuring the root logger.
"""

from __future__ import annotations

import logging
from typing import Optional

_CONSOLE_HANDLER_NAME = "sqs_pd_console"
_DEFAULT_FORMAT = "%(message)s"


def get_logger(name: str = "sqs_pd") -> logging.Logger:
    return logging.getLogger(name)


def ensure_console_handler(
    logger: logging.Logger,
    *,
    enabled: bool,
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
) -> None:
    """Attach/remove a StreamHandler to `logger` without touching root logging."""

    existing: Optional[logging.Handler] = None
    for h in logger.handlers:
        if getattr(h, "name", None) == _CONSOLE_HANDLER_NAME:
            existing = h
            break

    if not enabled:
        if existing is not None:
            logger.removeHandler(existing)
        return

    if existing is None:
        handler = logging.StreamHandler()
        handler.name = _CONSOLE_HANDLER_NAME
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    else:
        existing.setFormatter(logging.Formatter(fmt))

    logger.setLevel(min(logger.level or level, level))
    # Prevent double-printing if user configured root handlers.
    logger.propagate = False
