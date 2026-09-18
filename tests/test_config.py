from __future__ import annotations

import pytest

from app import config
from app.config import (
    DEFAULT_GEMINI_FALLBACK_MODELS,
    DEFAULT_GEMINI_MODEL,
    MIN_LLM_TIMEOUT_SECONDS,
    Settings,
    get_settings,
)

ENV_VARS = ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "LLM_TIMEOUT_SECONDS",
            "LLM_TOTAL_BUDGET_SECONDS", "LLM_HEDGE_DELAY_SECONDS", "LLM_MAX_RETRIES", "LLM_ENABLED",
            "LLM_CACHE_SIZE", "LOG_LEVEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def load(monkeypatch, **env) -> Settings:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    return get_settings()


def test_defaults(monkeypatch):
    s = load(monkeypatch)
    assert s.gemini_api_key is None
    assert s.gemini_model == DEFAULT_GEMINI_MODEL
    assert s.gemini_fallback_models == list(DEFAULT_GEMINI_FALLBACK_MODELS)
    assert s.model_chain == [DEFAULT_GEMINI_MODEL, *DEFAULT_GEMINI_FALLBACK_MODELS]
    assert (s.llm_timeout_seconds, s.llm_total_budget_seconds, s.llm_hedge_delay_seconds) == (10.0, 20.0, 2.5)
    assert s.llm_enabled and s.llm_max_retries == 1 and s.llm_cache_size == 512


def test_fallback_list_is_parsed_trimmed_and_deduplicated(monkeypatch):
    s = load(monkeypatch, GEMINI_FALLBACK_MODELS=" model-b , model-c,model-b,, ")
    assert s.gemini_fallback_models == ["model-b", "model-c"]


def test_empty_fallback_list_disables_fallbacks(monkeypatch):
    s = load(monkeypatch, GEMINI_FALLBACK_MODELS="")
    assert s.gemini_fallback_models == [] and s.model_chain == [DEFAULT_GEMINI_MODEL]


def test_primary_is_removed_from_the_fallback_list(monkeypatch):
    s = load(monkeypatch, GEMINI_MODEL="model-a", GEMINI_FALLBACK_MODELS="model-a,model-b")
    assert s.model_chain == ["model-a", "model-b"]


def test_blank_model_env_falls_back_to_the_default(monkeypatch):
    assert load(monkeypatch, GEMINI_MODEL="").gemini_model == DEFAULT_GEMINI_MODEL


def test_timeout_never_goes_below_the_api_minimum(monkeypatch):
    assert load(monkeypatch, LLM_TIMEOUT_SECONDS="8").llm_timeout_seconds == MIN_LLM_TIMEOUT_SECONDS
    assert load(monkeypatch, LLM_TIMEOUT_SECONDS="garbage").llm_timeout_seconds == MIN_LLM_TIMEOUT_SECONDS
    assert load(monkeypatch, LLM_TIMEOUT_SECONDS="15").llm_timeout_seconds == 15.0


def test_total_budget_is_clamped(monkeypatch):
    assert load(monkeypatch, LLM_TOTAL_BUDGET_SECONDS="1").llm_total_budget_seconds == config.MIN_LLM_TOTAL_BUDGET_SECONDS
    assert load(monkeypatch, LLM_TOTAL_BUDGET_SECONDS="999").llm_total_budget_seconds == config.MAX_LLM_TOTAL_BUDGET_SECONDS
    assert load(monkeypatch, LLM_TOTAL_BUDGET_SECONDS="12").llm_total_budget_seconds == 12.0


def test_hedge_delay_and_retries_cannot_go_negative(monkeypatch):
    s = load(monkeypatch, LLM_HEDGE_DELAY_SECONDS="-3", LLM_MAX_RETRIES="-2")
    assert s.llm_hedge_delay_seconds == 0.0 and s.llm_max_retries == 0
    assert load(monkeypatch, LLM_HEDGE_DELAY_SECONDS="0").llm_hedge_delay_seconds == 0.0


def test_llm_enabled_flag(monkeypatch):
    assert load(monkeypatch, LLM_ENABLED="false").llm_enabled is False
    assert load(monkeypatch, LLM_ENABLED="YES").llm_enabled is True


def test_settings_ship_without_a_default_secret():
    assert Settings().gemini_api_key is None
