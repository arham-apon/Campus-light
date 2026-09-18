"""Offline tests for the Gemini client: fallback chain, hedging, cooldowns, budget and cache.

A scripted fake replaces the SDK client, so nothing here touches the network. Sleeps are real but
short (<= ~0.6 s) so the concurrency behaviour is exercised for real.
"""
from __future__ import annotations

import threading
import time

import pytest

from app import llm_client as lc
from app.config import Settings
from app.schemas import DirectiveInterpretationPackage, LLMDirectiveEntry

PKG = DirectiveInterpretationPackage(entries=[LLMDirectiveEntry(note_index=0, directive_type="no_op")])
EMPTY = DirectiveInterpretationPackage(entries=[])


class Resp:
    def __init__(self, parsed=None, text=None):
        self.parsed, self.text = parsed, text


class ApiError(Exception):
    def __init__(self, code: int, message: str = ""):
        super().__init__(f"{code} {message}")
        self.code = code


def ok(delay: float = 0.0, package=PKG):
    def step():
        time.sleep(delay)
        return Resp(parsed=package)

    return step


def err(exc: BaseException, delay: float = 0.0):
    def step():
        time.sleep(delay)
        raise exc

    return step


class FakeClient:
    """`script[model]` is a list of steps consumed in order; the last one repeats."""

    def __init__(self, script: dict[str, list]):
        self.script = {m: list(steps) for m, steps in script.items()}
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self.models = self

    def generate_content(self, model, contents, config):
        with self._lock:
            self.calls.append(model)
            steps = self.script[model]
            step = steps.pop(0) if len(steps) > 1 else steps[0]
        return step()

    def get(self, model):
        return object()

    def count(self, model: str) -> int:
        return self.calls.count(model)


def settings(**overrides) -> Settings:
    base = dict(gemini_api_key="fake", gemini_model="P", gemini_fallback_models=["F1", "F2"],
                llm_hedge_delay_seconds=0.1, llm_total_budget_seconds=5.0, llm_max_retries=1, llm_cache_size=8)
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    lc._cache.clear()
    lc._cooldown_until.clear()
    lc._config_cache.clear()
    monkeypatch.setattr(lc, "_build_config", lambda model: object())  # skip the SDK import
    yield
    # Abandoned hedge/timeout calls keep running in worker threads until their fake sleep ends.
    # Wait for them so a straggler cannot update model health inside a later test.
    give_up = time.monotonic() + 3
    while time.monotonic() < give_up and any(t.name.startswith("llm") and t.is_alive() for t in threading.enumerate()):
        time.sleep(0.02)
    lc._cache.clear()
    lc._cooldown_until.clear()


def run(monkeypatch, script, notes=("a",), cooling=None, **overrides):
    """Extract with a fake client; returns (result, client). `cooling` maps model -> seconds left."""
    client = FakeClient(script)
    for model, seconds in (cooling or {}).items():
        lc._cooldown_until[model] = lc._now() + seconds
    monkeypatch.setattr(lc, "get_settings", lambda: settings(**overrides))
    monkeypatch.setattr(lc, "_get_client", lambda: client)
    return lc.extract_directives(list(notes), 200.0), client


# --- happy path and caching -----------------------------------------------------------------


def test_primary_answers_and_fallbacks_stay_idle(monkeypatch):
    result, client = run(monkeypatch, {"P": [ok()], "F1": [ok()], "F2": [ok()]})
    assert (result.source, result.model, result.attempts, result.error) == ("gemini", "P", 1, None)
    assert client.calls == ["P"]


def test_identical_notes_are_served_from_cache(monkeypatch):
    first, client = run(monkeypatch, {"P": [ok()]})
    second = lc.extract_directives(["a"], 200.0)
    assert (first.source, second.source) == ("gemini", "cache") and client.calls == ["P"]


def test_missing_key_reports_unavailable(monkeypatch):
    monkeypatch.setattr(lc, "get_settings", lambda: settings(gemini_api_key=None))
    monkeypatch.setattr(lc, "_client", None)
    result = lc.extract_directives(["a"], 200.0)
    assert (result.source, result.entries, result.error) == ("unavailable", [], "llm_unavailable")


# --- failover ---------------------------------------------------------------------------------


def test_rate_limited_primary_fails_over_immediately(monkeypatch):
    started = time.perf_counter()
    result, client = run(monkeypatch, {"P": [err(ApiError(429, "RESOURCE_EXHAUSTED 'retryDelay': '42s'"))],
                                       "F1": [ok()], "F2": [ok()]})
    assert (result.source, result.model, result.attempts) == ("gemini", "F1", 2)
    assert client.calls == ["P", "F1"]
    assert time.perf_counter() - started < 0.5  # no hedge wait: failure is acted on at once


def test_rate_limited_model_is_skipped_on_the_next_request(monkeypatch):
    _, client = run(monkeypatch, {"P": [err(ApiError(429, "'retryDelay': '42s'"))], "F1": [ok()]},
                    gemini_fallback_models=["F1"])
    lc.extract_directives(["different notes"], 200.0)
    assert client.calls == ["P", "F1", "F1"]  # the second request never touches the cooling primary
    assert 30 < lc._cooldown_until["P"] - lc._now() <= 42


def test_cooling_model_is_retried_once_its_cooldown_expires(monkeypatch):
    _, client = run(monkeypatch, {"P": [err(ApiError(429)), ok()], "F1": [ok()]}, gemini_fallback_models=["F1"])
    lc._cooldown_until["P"] = lc._now() - 1  # cooldown over
    result = lc.extract_directives(["later"], 200.0)
    assert result.model == "P"


def test_permanent_error_retires_the_model_for_a_long_while(monkeypatch):
    result, client = run(monkeypatch, {"P": [err(ApiError(404, "no longer available"))], "F1": [ok()]},
                         gemini_fallback_models=["F1"])
    assert result.model == "F1"
    assert lc._cooldown_until["P"] - lc._now() > 250
    lc.extract_directives(["more"], 200.0)
    assert client.count("P") == 1  # never tried again


def test_empty_answer_counts_as_a_failure(monkeypatch):
    result, client = run(monkeypatch, {"P": [ok(package=EMPTY)], "F1": [ok()]}, gemini_fallback_models=["F1"])
    assert (result.model, client.calls) == ("F1", ["P", "F1"])


def test_unparseable_answer_counts_as_a_failure(monkeypatch):
    def junk():
        return Resp(text="I cannot help with that")

    result, _ = run(monkeypatch, {"P": [junk], "F1": [ok()]}, gemini_fallback_models=["F1"])
    assert result.model == "F1"


def test_all_models_failing_degrades_without_raising(monkeypatch):
    result, client = run(monkeypatch, {m: [err(RuntimeError("down"))] for m in ("P", "F1", "F2")})
    assert (result.source, result.entries, result.error) == ("failed", [], "RuntimeError")
    assert result.attempts == 6 and client.count("P") == 2  # every model tried in both rounds


def test_retry_round_can_recover(monkeypatch):
    result, client = run(monkeypatch, {"P": [err(RuntimeError("blip")), ok()], "F1": [err(RuntimeError("blip"))]},
                         gemini_fallback_models=["F1"])
    assert result.source == "gemini" and result.attempts == 3


def test_retries_disabled(monkeypatch):
    result, client = run(monkeypatch, {"P": [err(RuntimeError("x"))]}, gemini_fallback_models=[], llm_max_retries=0)
    assert result.source == "failed" and client.calls == ["P"]


# --- hedging ----------------------------------------------------------------------------------


def test_slow_primary_is_hedged_and_the_fast_fallback_wins(monkeypatch):
    started = time.perf_counter()
    result, client = run(monkeypatch, {"P": [ok(delay=0.8)], "F1": [ok()]}, gemini_fallback_models=["F1"],
                         llm_hedge_delay_seconds=0.1)
    assert result.model == "F1" and result.attempts == 2
    assert time.perf_counter() - started < 0.5  # did not wait for the slow primary
    assert "P" not in lc._cooldown_until        # slow is not a failure


def test_fast_primary_is_not_hedged(monkeypatch):
    result, client = run(monkeypatch, {"P": [ok(delay=0.05)], "F1": [ok()]}, gemini_fallback_models=["F1"],
                         llm_hedge_delay_seconds=0.3)
    assert (result.model, client.calls) == ("P", ["P"])


def test_hedging_can_be_disabled(monkeypatch):
    result, client = run(monkeypatch, {"P": [ok(delay=0.3)], "F1": [ok()]}, gemini_fallback_models=["F1"],
                         llm_hedge_delay_seconds=0)
    assert (result.model, client.calls) == ("P", ["P"])


def test_a_failed_hedge_does_not_immediately_start_yet_another_call(monkeypatch):
    # P is slow but succeeds at 0.45 s. F1 is hedged in at 0.3 s and fails at once. Failing over to F2
    # right then would waste a call while P is still in flight; the next hedge would only fire at 0.6 s.
    result, client = run(monkeypatch, {"P": [ok(delay=0.45)], "F1": [err(ApiError(503))], "F2": [ok()]},
                         llm_hedge_delay_seconds=0.3)
    assert result.model == "P"
    assert client.calls == ["P", "F1"]


def test_failover_starts_the_next_model_the_moment_nothing_is_in_flight(monkeypatch):
    # P fails at 0.35 s while F1 (hedged in at 0.25 s) already failed: F2 starts at 0.35 s, not at
    # the next hedge tick (0.5 s).
    started = time.perf_counter()
    result, client = run(monkeypatch, {"P": [err(RuntimeError("late"), delay=0.35)],
                                       "F1": [err(RuntimeError("fast"))], "F2": [ok()]},
                         llm_hedge_delay_seconds=0.25)
    assert result.model == "F2" and client.calls == ["P", "F1", "F2"]
    assert time.perf_counter() - started < 0.47


def test_hedging_skips_models_known_to_be_cooling(monkeypatch):
    # P is slow. F1 is rate-limited (cooling), F2 is healthy: the hedge must go straight to F2.
    result, client = run(monkeypatch, {"P": [ok(delay=0.5)], "F1": [ok()], "F2": [ok()]},
                         cooling={"F1": 30}, llm_hedge_delay_seconds=0.1)
    assert result.model == "F2" and client.calls == ["P", "F2"]


def test_cooling_models_are_still_the_last_resort_on_failure(monkeypatch):
    # Nothing healthy is left after P fails, so the cooling F1 gets its chance (its quota may have recovered).
    result, client = run(monkeypatch, {"P": [err(ApiError(503))], "F1": [ok()]}, cooling={"F1": 30},
                         gemini_fallback_models=["F1"])
    assert result.model == "F1" and client.calls == ["P", "F1"]


def test_all_models_cooling_still_tries_the_soonest_to_recover(monkeypatch):
    result, client = run(monkeypatch, {"P": [ok()], "F1": [ok()]}, cooling={"P": 40, "F1": 10},
                         gemini_fallback_models=["F1"])
    assert result.model == "F1" and client.calls == ["F1"]


# --- budget -----------------------------------------------------------------------------------


def test_total_budget_bounds_the_wait(monkeypatch):
    started = time.perf_counter()
    result, _ = run(monkeypatch, {"P": [ok(delay=0.9)]}, gemini_fallback_models=[], llm_total_budget_seconds=0.4,
                    llm_max_retries=0)
    elapsed = time.perf_counter() - started
    assert result.source == "failed" and result.error == "BudgetExceeded"
    assert 0.35 < elapsed < 1.0


def test_no_extra_attempts_are_started_when_the_budget_is_nearly_spent(monkeypatch):
    # Budget below the minimum useful attempt time: only the first call is ever made.
    result, client = run(monkeypatch, {"P": [err(ApiError(503))], "F1": [ok()]}, gemini_fallback_models=["F1"],
                         llm_total_budget_seconds=1.0)
    assert result.source == "failed" and client.calls == ["P"]


# --- health bookkeeping -----------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("429 RESOURCE_EXHAUSTED {'retryDelay': '42s'}", 42.0),
    ("quota exceeded. Please retry in 41.9s.", 41.9),
    ("retryDelay: \"7s\"", 7.0),
    ("503 UNAVAILABLE", None),
])
def test_retry_hint_parsing(text, expected):
    assert lc._retry_delay_seconds(Exception(text)) == expected


@pytest.mark.parametrize("exc,expected", [
    (ApiError(429, "'retryDelay': '42s'"), 42.0),
    (ApiError(429, "'retryDelay': '900s'"), 60.0),    # capped at one quota window
    (ApiError(429, "'retryDelay': '0.2s'"), 1.0),     # floored
    (ApiError(429), 30.0),
    (ApiError(404), 300.0), (ApiError(400), 300.0), (ApiError(403), 300.0),
    (ApiError(503), 5.0), (ApiError(504), 5.0), (ApiError(500), 5.0), (ApiError(408), 5.0),
    (ValueError("bad json"), 30.0),
    (TimeoutError("read timed out"), 5.0),
])
def test_cooldown_lengths(exc, expected):
    assert lc._cooldown_seconds(exc) == expected


def test_daily_quota_gets_a_long_cooldown_but_per_minute_quota_does_not():
    per_day = ApiError(429, "{'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20', "
                            "'retryDelay': '37s'}")
    per_minute = ApiError(429, "{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', "
                               "'quotaValue': '15', 'retryDelay': '42s'}")
    assert lc._cooldown_seconds(per_day) == 3600.0   # the 37 s hint is not the real reset
    assert lc._cooldown_seconds(per_minute) == 42.0


def test_plan_puts_cooling_models_last_soonest_first(monkeypatch):
    now = lc._now()
    lc._cooldown_until.update({"P": now + 40, "F2": now + 10})
    assert lc._plan(["P", "F1", "F2"]) == ["F1", "F2", "P"]
    lc._cooldown_until.clear()
    assert lc._plan(["P", "F1", "F2"]) == ["P", "F1", "F2"]


def test_success_clears_a_cooldown(monkeypatch):
    lc._cooldown_until["P"] = lc._now() - 1
    run(monkeypatch, {"P": [ok()]}, gemini_fallback_models=[])
    assert "P" not in lc._cooldown_until


# --- warm-up ----------------------------------------------------------------------------------


def test_warmup_never_raises(monkeypatch):
    class Boom(FakeClient):
        def get(self, model):
            raise ApiError(404, "no such model")

    monkeypatch.setattr(lc, "get_settings", lambda: settings())
    monkeypatch.setattr(lc, "_get_client", lambda: Boom({}))
    lc.warmup()
    monkeypatch.setattr(lc, "_get_client", lambda: (_ for _ in ()).throw(RuntimeError("no sdk")))
    lc.warmup()


def test_warmup_async_returns_a_daemon_thread(monkeypatch):
    monkeypatch.setattr(lc, "get_settings", lambda: settings())
    monkeypatch.setattr(lc, "_get_client", lambda: FakeClient({}))
    thread = lc.warmup_async()
    thread.join(timeout=2)
    assert thread.daemon and not thread.is_alive()


# --- per-model request config -----------------------------------------------------------------


@pytest.mark.parametrize("model,expected", [
    ("gemini-3.5-flash", True), ("gemini-3-flash-preview", True), ("gemini-3.8-flash", True),
    ("gemini-3.5-flash-lite", False), ("gemini-3.1-flash-lite", False), ("gemini-3.1-pro-preview", False),
    ("gemma-4-26b-a4b-it", False), ("gemini-2.5-flash", False), ("gemini-flash-latest", False),
])
def test_only_plain_gemini_3_flash_gets_minimal_thinking(model, expected):
    assert lc._wants_minimal_thinking(model) is expected


def test_real_config_is_accepted_by_the_installed_sdk(monkeypatch):
    monkeypatch.undo()  # the autouse fixture stubs _build_config; use the real one here
    lc._config_cache.clear()
    thinking = lc._build_config("gemini-3.5-flash")
    plain = lc._build_config("gemini-3.5-flash-lite")
    assert thinking.thinking_config is not None and str(thinking.thinking_config.thinking_level).lower().endswith("minimal")
    assert plain.thinking_config is None
    assert thinking.temperature == 0.0 and thinking.response_mime_type == "application/json"
    assert lc._build_config("gemini-3.5-flash") is thinking  # cached
