"""Gemini structured extraction: model fallback chain, hedged requests, cache and safe degradation.

Request flow for one set of operator notes:

1. Identical notes seen before are served from an in-process LRU cache.
2. Otherwise the models in ``Settings.model_chain`` are tried in order. The next model starts as soon
   as the current one fails, or in parallel when it has not answered after ``llm_hedge_delay_seconds``.
   The first usable answer wins; a model that is rate-limited or down is skipped for a while.
3. Nothing here ever raises. If no model answers within ``llm_total_budget_seconds`` the result has
   no entries and the guardrails turn every note into ``no_op``.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.logging_utils import get_logger
from app.prompts import SYSTEM_INSTRUCTION, build_user_prompt
from app.schemas import DirectiveInterpretationPackage, LLMDirectiveEntry

log = get_logger(__name__)

_RETRYABLE_4XX = {408, 429}  # request timeout, quota; every other 4xx is permanent for that model

# How long a model is skipped (moved to the back of the line) after a failure.
_QUOTA_COOLDOWN_DEFAULT_SECONDS = 30.0   # 429 without a usable retry hint
_QUOTA_COOLDOWN_MAX_SECONDS = 60.0       # the free tier's per-minute window; re-probe at least this often
_PERMANENT_COOLDOWN_SECONDS = 300.0      # 400/401/403/404: bad config or retired model
_BAD_OUTPUT_COOLDOWN_SECONDS = 30.0      # unparseable / empty answers
_TRANSIENT_COOLDOWN_SECONDS = 5.0        # 5xx, timeouts, connection errors

# An extra attempt is only worth starting if at least this much of the budget is left.
_MIN_USEFUL_ATTEMPT_SECONDS = 3.0

_RETRY_DELAY_RE = re.compile(r"retry(?:Delay)?['\"]?\s*[:=]?\s*(?:in\s+)?['\"]?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)

_client: Any = None
_client_lock = threading.Lock()
_cache: "OrderedDict[tuple, list[LLMDirectiveEntry]]" = OrderedDict()
_cache_lock = threading.Lock()
_cooldown_until: dict[str, float] = {}
_health_lock = threading.Lock()
_config_cache: dict[str, Any] = {}


class ExtractionResult:
    """Carries the entries plus provenance for logging and diagnostics."""

    __slots__ = ("entries", "source", "latency_ms", "error", "model", "attempts")

    def __init__(
        self,
        entries: list[LLMDirectiveEntry],
        source: str,
        latency_ms: float,
        error: str | None = None,
        model: str | None = None,
        attempts: int = 0,
    ):
        self.entries = entries
        self.source = source
        self.latency_ms = latency_ms
        self.error = error
        self.model = model
        self.attempts = attempts


@dataclass
class _Outcome:
    """Result of one call to one model."""

    model: str
    entries: list[LLMDirectiveEntry] | None
    error: str | None = None
    status: int | None = None
    elapsed_ms: float = 0.0

    @property
    def permanent(self) -> bool:
        return self.status is not None and 400 <= self.status < 500 and self.status not in _RETRYABLE_4XX


@dataclass
class _Race:
    winner: _Outcome | None = None
    failures: list[_Outcome] = field(default_factory=list)
    launched: int = 0


def _now() -> float:
    return time.monotonic()


# --------------------------------------------------------------------------
# Client and warm-up
# --------------------------------------------------------------------------


def _get_client():
    """Lazily build one shared client. Returns None when no key is configured."""
    global _client
    if _client is not None:
        return _client
    settings = get_settings()
    if not settings.gemini_api_key or not settings.llm_enabled:
        return None
    with _client_lock:
        if _client is None:
            from google import genai
            from google.genai import types

            timeout_ms = int(settings.llm_timeout_seconds * 1000)
            try:
                _client = genai.Client(
                    api_key=settings.gemini_api_key,
                    http_options=types.HttpOptions(timeout=timeout_ms),
                )
            except TypeError:  # older SDK without http_options
                _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def warmup() -> None:
    """Pay the one-off costs at startup: SDK import, client build and the first TLS handshake.

    Measured on a cold process, the import (~1.3 s), client build (~0.4 s) and handshake (~0.5 s)
    would otherwise land on the first judged request. Never raises and never needs the model to work.
    """
    try:
        client = _get_client()
        if client is None:
            return
        settings = get_settings()
        for model in settings.model_chain:
            _build_config(model)
        client.models.get(model=settings.gemini_model)  # metadata call: opens the pooled connection
        log.info("llm warmup complete model=%s", settings.gemini_model)
    except Exception as exc:  # never block startup
        log.warning("llm warmup skipped: %s status=%s", type(exc).__name__, _status_of(exc))


def warmup_async() -> threading.Thread:
    """Run warmup() on a daemon thread so startup and /health never wait for the network."""
    thread = threading.Thread(target=warmup, name="llm-warmup", daemon=True)
    thread.start()
    return thread


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _cache_get(key: tuple) -> list[LLMDirectiveEntry] | None:
    with _cache_lock:
        entries = _cache.get(key)
        if entries is not None:
            _cache.move_to_end(key)
        return entries


def _cache_put(key: tuple, entries: list[LLMDirectiveEntry]) -> None:
    limit = get_settings().llm_cache_size
    if limit <= 0:
        return
    with _cache_lock:
        _cache[key] = entries
        _cache.move_to_end(key)
        while len(_cache) > limit:
            _cache.popitem(last=False)


# --------------------------------------------------------------------------
# Per-model health: cooldowns
# --------------------------------------------------------------------------


def _status_of(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _retry_delay_seconds(exc: BaseException) -> float | None:
    """Parse the provider's retry hint, e.g. "'retryDelay': '42s'" or "Please retry in 41.9s"."""
    match = _RETRY_DELAY_RE.search(str(exc))
    return float(match.group(1)) if match else None


def _cooldown_seconds(exc: BaseException) -> float:
    status = _status_of(exc)
    if status == 429:
        hint = _retry_delay_seconds(exc)
        return min(_QUOTA_COOLDOWN_MAX_SECONDS, max(1.0, hint)) if hint else _QUOTA_COOLDOWN_DEFAULT_SECONDS
    if status is not None:
        if 400 <= status < 500 and status != 408:
            return _PERMANENT_COOLDOWN_SECONDS
        return _TRANSIENT_COOLDOWN_SECONDS
    if isinstance(exc, ValueError):  # JSON / schema / empty answer
        return _BAD_OUTPUT_COOLDOWN_SECONDS
    return _TRANSIENT_COOLDOWN_SECONDS  # timeouts, connection errors


def _mark_failure(model: str, exc: BaseException) -> None:
    until = _now() + _cooldown_seconds(exc)
    with _health_lock:
        _cooldown_until[model] = max(_cooldown_until.get(model, 0.0), until)


def _mark_success(model: str) -> None:
    with _health_lock:
        _cooldown_until.pop(model, None)


def _plan(models: list[str]) -> list[str]:
    """Healthy models in configured order, then cooling ones, soonest to recover first."""
    now = _now()
    with _health_lock:
        until = {m: _cooldown_until.get(m, 0.0) for m in models}
    order = {m: i for i, m in enumerate(models)}
    return sorted(models, key=lambda m: (until[m] > now, until[m] if until[m] > now else 0.0, order[m]))


# --------------------------------------------------------------------------
# One call to one model
# --------------------------------------------------------------------------


def _build_config(model: str):
    cached = _config_cache.get(model)
    if cached is not None:
        return cached
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
        top_p=1.0,
        candidate_count=1,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=DirectiveInterpretationPackage,
    )
    _config_cache[model] = config
    return config


def _parse_response(response: Any) -> list[LLMDirectiveEntry]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, DirectiveInterpretationPackage):
        return list(parsed.entries)
    if isinstance(parsed, dict):
        return list(DirectiveInterpretationPackage.model_validate(parsed).entries)

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ValueError("empty model response")
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model response")
    payload = json.loads(text[start : end + 1])
    return list(DirectiveInterpretationPackage.model_validate(payload).entries)


def _attempt(model: str, prompt: str) -> _Outcome:
    """Call one model. Never raises: failures come back as an _Outcome and update the model's health."""
    started = time.perf_counter()
    try:
        client = _get_client()
        if client is None:
            raise RuntimeError("llm client unavailable")
        response = client.models.generate_content(model=model, contents=prompt, config=_build_config(model))
        entries = _parse_response(response)
        if not entries:
            raise ValueError("model returned no entries")
        _mark_success(model)
        return _Outcome(model, entries, elapsed_ms=(time.perf_counter() - started) * 1000)
    except Exception as exc:  # noqa: BLE001 - provider errors must never escape
        _mark_failure(model, exc)
        status = _status_of(exc)
        elapsed = (time.perf_counter() - started) * 1000
        log.warning("llm attempt failed model=%s error=%s status=%s %.0fms", model, type(exc).__name__, status, elapsed)
        return _Outcome(model, None, type(exc).__name__, status, elapsed)


# --------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------


def _race(models: list[str], prompt: str, hedge_delay: float, deadline: float) -> _Race:
    """Try `models` in order until one answers or `deadline` (a _now() value) passes.

    The next model starts when the current one fails, or alongside it once `hedge_delay` seconds
    pass without an answer (0 disables that). While attempts are in flight a failure does not start
    another one, so each request spends at most as many calls as it needs.
    """
    race = _Race()
    if not models:
        return race

    pool = ThreadPoolExecutor(max_workers=len(models), thread_name_prefix="llm")
    pending: dict[Future, str] = {}
    queue = list(models)

    def launch() -> bool:
        if not queue:
            return False
        if race.launched and deadline - _now() < _MIN_USEFUL_ATTEMPT_SECONDS:
            return False
        model = queue.pop(0)
        pending[pool.submit(_attempt, model, prompt)] = model
        race.launched += 1
        return True

    try:
        launch()
        while pending:
            remaining = deadline - _now()
            if remaining <= 0:
                break
            hedging = hedge_delay > 0 and bool(queue) and remaining >= _MIN_USEFUL_ATTEMPT_SECONDS
            done, _ = wait(list(pending), timeout=min(remaining, hedge_delay) if hedging else remaining,
                           return_when=FIRST_COMPLETED)
            if not done:
                if hedging:
                    launch()  # the current model is slow: start the next one alongside it
                continue
            for future in done:
                pending.pop(future)
                outcome = future.result()
                if outcome.entries is not None:
                    race.winner = outcome
                    return race
                race.failures.append(outcome)
            if not pending:
                launch()  # everything in flight failed: fail over to the next model
        for future, model in pending.items():  # deadline hit with calls still running
            race.failures.append(_Outcome(model, None, "BudgetExceeded"))
        return race
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def extract_directives(notes: list[str], battery_capacity_kwh: float) -> ExtractionResult:
    """Interpret operator notes with Gemini. Never raises; degrades to an empty list."""
    started = time.perf_counter()
    key = (tuple(n.strip() for n in notes), round(float(battery_capacity_kwh), 6))

    cached = _cache_get(key)
    if cached is not None:
        return ExtractionResult(list(cached), "cache", (time.perf_counter() - started) * 1000)

    if _get_client() is None:
        log.error("LLM unavailable: GEMINI_API_KEY missing or LLM_ENABLED=false")
        return ExtractionResult([], "unavailable", (time.perf_counter() - started) * 1000, "llm_unavailable")

    settings = get_settings()
    prompt = build_user_prompt(notes, battery_capacity_kwh)
    deadline = _now() + settings.llm_total_budget_seconds
    dead: set[str] = set()  # models that failed permanently during this request
    attempts = 0
    last_error: str | None = None

    for round_no in range(settings.llm_max_retries + 1):
        models = [m for m in _plan(settings.model_chain) if m not in dead]
        if not models:
            break
        if round_no > 0:
            backoff = 0.25 * round_no
            if deadline - _now() < backoff + _MIN_USEFUL_ATTEMPT_SECONDS:
                break
            time.sleep(backoff)

        race = _race(models, prompt, settings.llm_hedge_delay_seconds, deadline)
        attempts += race.launched
        for failure in race.failures:
            last_error = failure.error
            if failure.permanent:
                dead.add(failure.model)

        if race.winner is not None:
            entries = race.winner.entries or []
            _cache_put(key, list(entries))
            elapsed = (time.perf_counter() - started) * 1000
            log.info("llm extraction ok model=%s notes=%d entries=%d attempts=%d %.0fms",
                     race.winner.model, len(notes), len(entries), attempts, elapsed)
            return ExtractionResult(entries, "gemini", elapsed, model=race.winner.model, attempts=attempts)

        if deadline - _now() < _MIN_USEFUL_ATTEMPT_SECONDS:
            break

    elapsed = (time.perf_counter() - started) * 1000
    log.error("llm extraction failed after %d attempt(s) (%s) %.0fms", attempts, last_error, elapsed)
    return ExtractionResult([], "failed", elapsed, last_error, attempts=attempts)
