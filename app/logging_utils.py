"""Logger that can never print a credential."""
from __future__ import annotations

import logging
import os
import re
import sys

_REDACTIONS: list[re.Pattern[str]] = [
    re.compile(r"AIza[0-9A-Za-z\-_]{10,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*\S+"),
]


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        key = os.getenv("GEMINI_API_KEY")
        if key:
            msg = msg.replace(key, "***redacted***")
        for pattern in _REDACTIONS:
            msg = pattern.sub("***redacted***", msg)
        return msg


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
