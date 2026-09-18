"""Test fixtures. Default mode stubs the LLM so the suite runs with no API key."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import llm_client
from app.llm_client import ExtractionResult
from app.schemas import LLMDirectiveEntry

SAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


@pytest.fixture(scope="session")
def sample_pack() -> dict:
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))


def _expected_to_llm_entries(expected: list[dict]) -> list[LLMDirectiveEntry]:
    """Turn a case's ground-truth interpretation into the flat shape the model would emit."""
    entries = []
    for item in expected:
        adjustment = item.get("structured_adjustment") or {}
        entries.append(
            LLMDirectiveEntry(
                note_index=item["note_index"],
                directive_type=item["directive_type"],
                hours=adjustment.get("hours", []),
                factor=adjustment.get("factor"),
                minimum_energy_kwh=adjustment.get("minimum_energy_kwh"),
                max_grid_kwh=adjustment.get("max_grid_kwh"),
                explanation=item.get("explanation", ""),
            )
        )
    return entries


@pytest.fixture
def client(monkeypatch, sample_pack) -> TestClient:
    """TestClient whose LLM step is replaced by ground truth, unless RUN_LLM_TESTS=1."""
    if os.getenv("RUN_LLM_TESTS") != "1":
        by_notes = {
            tuple(n.strip() for n in case["input"]["operator_notes"]):
            _expected_to_llm_entries(case["expected_output"]["directive_interpretation"])
            for case in sample_pack["cases"]
        }

        def fake_extract(notes, battery_capacity_kwh):
            key = tuple(n.strip() for n in notes)
            return ExtractionResult(list(by_notes.get(key, [])), "stub", 0.0)

        monkeypatch.setattr(llm_client, "extract_directives", fake_extract)
        # Offline mode must never open a network connection, so skip the startup warm-up too.
        monkeypatch.setattr(llm_client, "warmup_async", lambda: None)

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
