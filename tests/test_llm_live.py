"""Real Gemini calls. Enable with RUN_LLM_TESTS=1 and a valid GEMINI_API_KEY."""
from __future__ import annotations

import os

import pytest

from app.guardrails import apply_deterministic_guardrails
from app.llm_client import extract_directives

pytestmark = pytest.mark.skipif(os.getenv("RUN_LLM_TESTS") != "1", reason="live LLM tests are opt-in")


def test_every_public_case_extracts_ground_truth(sample_pack):
    failures = []
    for case in sample_pack["cases"]:
        notes = case["input"]["operator_notes"]
        capacity = case["input"]["battery"]["capacity_kwh"]
        result = extract_directives(notes, capacity)
        entries = apply_deterministic_guardrails(result.entries, len(notes), capacity,
                                                 case["input"]["battery"]["minimum_energy_kwh"])
        for got, want in zip(entries, case["expected_output"]["directive_interpretation"]):
            if got.directive_type.value != want["directive_type"]:
                failures.append(f"{case['id']} note {got.note_index}: type {got.directive_type.value} != {want['directive_type']}")
                continue
            if want["structured_adjustment"] is None:
                continue
            produced = got.structured_adjustment.model_dump()
            if produced.get("hours") != want["structured_adjustment"]["hours"]:
                failures.append(f"{case['id']} note {got.note_index}: hours {produced.get('hours')} != {want['structured_adjustment']['hours']}")
            for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
                if key in want["structured_adjustment"]:
                    if abs(produced.get(key, -1) - want["structured_adjustment"][key]) > 0.01:
                        failures.append(f"{case['id']} note {got.note_index}: {key} {produced.get(key)} != {want['structured_adjustment'][key]}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("note,expected_hours,expected_factor", [
    ("PV production will drop to about 20% between 13:00 and 15:00.", [13, 14], 0.2),
    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.", [13, 14], 0.2),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.", [13, 14], 0.2),
])
def test_paraphrase_robustness(note, expected_hours, expected_factor):
    entries = apply_deterministic_guardrails(extract_directives([note], 200.0).entries, 1, 200.0, 40.0)
    adjustment = entries[0].structured_adjustment.model_dump()
    assert entries[0].directive_type.value == "solar_reduction"
    assert adjustment["hours"] == expected_hours
    assert abs(adjustment["factor"] - expected_factor) <= 0.01
