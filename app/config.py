"""Environment-driven settings. No secret ever has a default value."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=False)


class Settings(BaseModel):
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    llm_timeout_seconds: float = 8.0
    llm_max_retries: int = 1
    llm_enabled: bool = True
    llm_cache_size: int = 512
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    def _f(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except ValueError:
            return default

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        llm_timeout_seconds=_f("LLM_TIMEOUT_SECONDS", 8.0),
        llm_max_retries=int(_f("LLM_MAX_RETRIES", 1)),
        llm_enabled=os.getenv("LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes"},
        llm_cache_size=int(_f("LLM_CACHE_SIZE", 512)),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
