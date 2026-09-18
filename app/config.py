"""Environment-driven settings. No secret ever has a default value."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(override=False)

# gemini-2.0-flash (the original default) now returns 404, and the 2.5 family is closed to
# new keys. gemini-3.5-flash-lite scored 18/18 on the public notes at ~1.3 s per call.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Tried when the primary is rate-limited, overloaded, slow or gone. Free-tier quotas are enforced
# per model (15 requests/minute each), so every extra model also adds request capacity.
DEFAULT_GEMINI_FALLBACK_MODELS: tuple[str, ...] = ("gemini-3.5-flash", "gemini-3.1-flash-lite")

# The Gemini API rejects any request deadline under 10 s with a 400 ("Minimum allowed deadline
# is 10s"), which would silently turn every note into no_op.
MIN_LLM_TIMEOUT_SECONDS = 10.0

# After this long without a usable answer a request gives up on the LLM (notes degrade to no_op),
# whatever the per-call timeouts say. Keeps the whole request well inside the 30 s hard limit.
DEFAULT_LLM_TOTAL_BUDGET_SECONDS = 20.0
MIN_LLM_TOTAL_BUDGET_SECONDS = 5.0
MAX_LLM_TOTAL_BUDGET_SECONDS = 25.0

# If the current model has not answered after this long, the next model is started in parallel
# and the first usable answer wins. Warm calls take ~1.3 s, so this only fires on real outliers.
# 0 disables hedging (fallbacks are then used only after a failure).
DEFAULT_LLM_HEDGE_DELAY_SECONDS = 2.5


class Settings(BaseModel):
    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_fallback_models: list[str] = Field(default_factory=lambda: list(DEFAULT_GEMINI_FALLBACK_MODELS))
    llm_timeout_seconds: float = MIN_LLM_TIMEOUT_SECONDS
    llm_total_budget_seconds: float = DEFAULT_LLM_TOTAL_BUDGET_SECONDS
    llm_hedge_delay_seconds: float = DEFAULT_LLM_HEDGE_DELAY_SECONDS
    llm_max_retries: int = 1
    llm_enabled: bool = True
    llm_cache_size: int = 512
    log_level: str = "INFO"

    @property
    def model_chain(self) -> list[str]:
        """Primary first, then fallbacks in order, without duplicates."""
        return list(dict.fromkeys([self.gemini_model, *self.gemini_fallback_models]))


def _parse_model_list(raw: str | None, default: tuple[str, ...]) -> list[str]:
    """Unset -> defaults. Set (even to an empty string) -> exactly what was given."""
    if raw is None:
        return list(default)
    return [name.strip() for name in raw.split(",") if name.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    def _f(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except ValueError:
            return default

    primary = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    fallbacks = _parse_model_list(os.getenv("GEMINI_FALLBACK_MODELS"), DEFAULT_GEMINI_FALLBACK_MODELS)

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=primary,
        gemini_fallback_models=[m for m in dict.fromkeys(fallbacks) if m != primary],
        llm_timeout_seconds=max(MIN_LLM_TIMEOUT_SECONDS, _f("LLM_TIMEOUT_SECONDS", MIN_LLM_TIMEOUT_SECONDS)),
        llm_total_budget_seconds=min(
            MAX_LLM_TOTAL_BUDGET_SECONDS,
            max(MIN_LLM_TOTAL_BUDGET_SECONDS, _f("LLM_TOTAL_BUDGET_SECONDS", DEFAULT_LLM_TOTAL_BUDGET_SECONDS)),
        ),
        llm_hedge_delay_seconds=max(0.0, _f("LLM_HEDGE_DELAY_SECONDS", DEFAULT_LLM_HEDGE_DELAY_SECONDS)),
        llm_max_retries=max(0, int(_f("LLM_MAX_RETRIES", 1))),
        llm_enabled=os.getenv("LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes"},
        llm_cache_size=int(_f("LLM_CACHE_SIZE", 512)),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
