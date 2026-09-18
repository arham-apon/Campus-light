"""Environment-driven settings. No secret ever has a default value."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=False)

# gemini-2.0-flash (the original default) now returns 404, and the 2.5 family is closed to
# new keys. gemini-3.5-flash-lite scored 18/18 on the public notes at ~1.4 s per call.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# The Gemini API rejects any request deadline under 10 s with a 400 ("Minimum allowed deadline
# is 10s"), which would silently turn every note into no_op. Worst case with one retry stays
# ~2 x 10 s, inside the 30 s hard limit.
MIN_LLM_TIMEOUT_SECONDS = 10.0


class Settings(BaseModel):
    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    llm_timeout_seconds: float = MIN_LLM_TIMEOUT_SECONDS
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
        gemini_model=os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
        llm_timeout_seconds=max(MIN_LLM_TIMEOUT_SECONDS, _f("LLM_TIMEOUT_SECONDS", MIN_LLM_TIMEOUT_SECONDS)),
        llm_max_retries=int(_f("LLM_MAX_RETRIES", 1)),
        llm_enabled=os.getenv("LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes"},
        llm_cache_size=int(_f("LLM_CACHE_SIZE", 512)),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
