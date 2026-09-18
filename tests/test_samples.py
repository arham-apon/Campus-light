"""End-to-end verification against all ten public sample cases."""
from __future__ import annotations

import pytest

from tests.replay import compile_expected, replay

TOL = 0.01


def _ids(pack):
    return [case["id"] for case in pack["cases"]]


@pytest.fixture(params=range(10))
def case(request, sample_pack):
    return sample_pack["cases"][request.param]


def test_case_pack_is_complete(sample_pack):
    assert len(sample_pack["cases"]) == 10


def test_end_to_end(case, client):
    payload = case["input"]
    expected = case["expected_output"]

    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    # --- API contract -----------------------------------------------------
    assert body["scenario_id"] == payload["scenario_id"]
    for field in ("directive_interpretation", "hourly_plan", "total_grid_kwh",
                  "total_cost_bdt", "peak_grid_kwh", "plan_summary"):
        assert field in body, f"missing response field {field}"
    assert isinstance(body["plan_summary"], str) and body["plan_summary"].strip()
    assert len(body["hourly_plan"]) == 24

    # --- Interpretation mapping & guardrail integrity ---------------------
    produced = body["directive_interpretation"]
    truth = expected["directive_interpretation"]
    assert len(produced) == len(payload["operator_notes"])
    assert [e["note_index"] for e in produced] == list(range(len(truth)))

    for got, want in zip(produced, truth):
        assert got["directive_type"] == want["directive_type"]
        assert got["applies"] == want["applies"]
        assert got["applies"] == (got["directive_type"] != "no_op")
        assert isinstance(got.get("explanation", ""), str)
        if want["directive_type"] == "no_op":
            assert got["structured_adjustment"] is None
            continue

        adjustment = got["structured_adjustment"]
        want_adjustment = want["structured_adjustment"]
        assert adjustment is not None
        assert set(adjustment.keys()) == set(want_adjustment.keys()), "structured_adjustment shape mismatch"
        assert adjustment["hours"] == want_adjustment["hours"]
        assert adjustment["hours"] == sorted(set(adjustment["hours"]))
        assert all(0 <= h <= 23 for h in adjustment["hours"])
        for numeric in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if numeric in want_adjustment:
                assert abs(adjustment[numeric] - want_adjustment[numeric]) <= TOL, numeric
        if "factor" in adjustment:
            assert 0.0 <= adjustment["factor"] <= 1.0

    # --- Full constraint replay against ground-truth directives -----------
    base_solar = [h["solar_kwh"] for h in sorted(payload["hours"], key=lambda x: x["hour"])]
    compiled = compile_expected(truth, base_solar)
    errors = replay(body, payload, *compiled)
    assert not errors, f"{case['id']} replay failures: {errors}"

    # --- Optimization quality --------------------------------------------
    assert body["total_cost_bdt"] <= expected["total_cost_bdt"] + TOL, (
        f"{case['id']} cost {body['total_cost_bdt']} worse than reference optimum {expected['total_cost_bdt']}"
    )
    assert abs(body["total_cost_bdt"] - expected["total_cost_bdt"]) <= TOL
    assert abs(body["total_grid_kwh"] - expected["total_grid_kwh"]) <= TOL
    assert abs(body["peak_grid_kwh"] - expected["peak_grid_kwh"]) <= TOL
