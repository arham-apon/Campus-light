"""Gemini structured extraction with timeout, retry, cache and safe degradation."""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any

from app.config import get_settings
from app.logging_utils import get_logger
from app.prompts import SYSTEM_INSTRUCTION, build_user_prompt
from app.schemas import DirectiveInterpretationPackage, LLMDirectiveEntry

log = get_logger(__name__)

_RETRYABLE_4XX = {408, 429}  # request timeout, quota; every other 4xx is permanent

_client: Any = None
_client_lock = threading.Lock()
_cache: "OrderedDict[tuple, list[LLMDirectiveEntry]]" = OrderedDict()
_cache_lock = threading.Lock()


class ExtractionResult:
    """Carries the entries plus provenance for logging and diagnostics."""

    __slots__ = ("entries", "source", "latency_ms", "error")

    def __init__(self, entries: list[LLMDirectiveEntry], source: str, latency_ms: float, error: str | None = None):
        self.entries = entries
        self.source = source
        self.latency_ms = latency_ms
        self.error = error


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
    """Build the client at startup so the first judged request pays no import cost."""
    try:
        _get_client()
    except Exception as exc:  # never block startup
        log.warning("llm warmup skipped: %s", type(exc).__name__)


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


def extract_directives(notes: list[str], battery_capacity_kwh: float) -> ExtractionResult:
    """Interpret operator notes with Gemini. Never raises; degrades to an empty list."""
    started = time.perf_counter()
    key = (tuple(n.strip() for n in notes), round(float(battery_capacity_kwh), 6))

    cached = _cache_get(key)
    if cached is not None:
        return ExtractionResult(list(cached), "cache", (time.perf_counter() - started) * 1000)

    client = _get_client()
    if client is None:
        log.error("LLM unavailable: GEMINI_API_KEY missing or LLM_ENABLED=false")
        return ExtractionResult([], "unavailable", (time.perf_counter() - started) * 1000, "llm_unavailable")

    from google.genai import types

    settings = get_settings()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
        top_p=1.0,
        candidate_count=1,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=DirectiveInterpretationPackage,
    )
    prompt = build_user_prompt(notes, battery_capacity_kwh)

    last_error: str | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=config,
            )
            entries = _parse_response(response)
            _cache_put(key, list(entries))
            elapsed = (time.perf_counter() - started) * 1000
            log.info("llm extraction ok notes=%d entries=%d attempt=%d %.0fms", len(notes), len(entries), attempt, elapsed)
            return ExtractionResult(entries, "gemini", elapsed)
        except Exception as exc:  # noqa: BLE001 - provider errors must never escape
            last_error = type(exc).__name__
            status = getattr(exc, "code", None)
            log.warning("llm extraction attempt %d failed: %s status=%s", attempt, last_error, status)
            if isinstance(status, int) and 400 <= status < 500 and status not in _RETRYABLE_4XX:
                break  # permanent client error (bad request, auth, unknown model): a retry cannot succeed
            if attempt < settings.llm_max_retries:
                time.sleep(0.25 * (attempt + 1))

    elapsed = (time.perf_counter() - started) * 1000
    log.error("llm extraction exhausted retries (%s) after %.0fms", last_error, elapsed)
    return ExtractionResult([], "failed", elapsed, last_error)
