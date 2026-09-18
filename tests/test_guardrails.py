from __future__ import annotations

import json

from app.guardrails import MAX_EXPLANATION_CHARS, apply_deterministic_guardrails as guard
from app.schemas import LLMDirectiveEntry as E


def test_fills_missing_entries_with_no_op():
    out = guard([], 3, 200.0, 40.0)
    assert [e.note_index for e in out] == [0, 1, 2]
    assert all(e.directive_type.value == "no_op" and not e.applies and e.structured_adjustment is None for e in out)


def test_hours_are_deduped_sorted_and_clamped():
    out = guard([E(note_index=0, directive_type="no_charge_window", hours=[15, 14, 14, 99, -3])], 1, 200.0, 40.0)
    assert out[0].structured_adjustment.model_dump() == {"hours": [14, 15]}


def test_percentage_factor_is_rescaled_and_clamped():
    out = guard([E(note_index=0, directive_type="solar_reduction", hours=[12], factor=25)], 1, 200.0, 40.0)
    assert out[0].structured_adjustment.model_dump()["factor"] == 0.25
    out = guard([E(note_index=0, directive_type="solar_reduction", hours=[12], factor=-4)], 1, 200.0, 40.0)
    assert out[0].structured_adjustment.model_dump()["factor"] == 0.0


def _factor(value):
    out = guard([E(note_index=0, directive_type="solar_reduction", hours=[12], factor=value)], 1, 200.0, 40.0)
    return out[0].structured_adjustment.model_dump()["factor"]


def test_factor_boundaries_match_the_failure_matrix():
    # Appendix C of the spec: 80 -> 0.8 (percentage rescue), 1.4 -> 1.0 (clamp)
    assert _factor(80) == 0.8
    assert _factor(1.4) == 1.0 and _factor(1.99) == 1.0
    assert _factor(2) == 0.02 and _factor(100) == 1.0 and _factor(100.5) == 1.0
    assert _factor(1.0) == 1.0 and _factor(0.5) == 0.5 and _factor(0.0) == 0.0
    assert _factor(-0.0) == 0.0 and _factor(-4) == 0.0


def test_flagged_reserve_boundaries():
    def reserve(value):
        out = guard([E(note_index=0, directive_type="minimum_battery_reserve", hours=[18], minimum_energy_kwh=value,
                       reserve_is_fraction_of_capacity=True)], 1, 200.0, 40.0)
        return out[0].structured_adjustment.model_dump()["minimum_energy_kwh"]

    assert reserve(0.75) == 150.0 and reserve(1.0) == 200.0
    assert reserve(1.5) == 200.0            # out-of-range fraction -> clamped to full capacity
    assert reserve(50) == 100.0 and reserve(100) == 200.0
    assert reserve(9999) == 200.0 and reserve(-3) == 0.0


def test_fractional_reserve_is_expanded_against_capacity():
    out = guard([E(note_index=0, directive_type="minimum_battery_reserve", hours=[18], minimum_energy_kwh=0.5,
                   reserve_is_fraction_of_capacity=True)], 1, 200.0, 40.0)
    assert out[0].structured_adjustment.model_dump()["minimum_energy_kwh"] == 100.0


def test_reserve_cannot_exceed_capacity():
    out = guard([E(note_index=0, directive_type="minimum_battery_reserve", hours=[18], minimum_energy_kwh=9999)], 1, 200.0, 40.0)
    assert out[0].structured_adjustment.model_dump()["minimum_energy_kwh"] == 200.0


def test_directive_without_hours_becomes_no_op():
    out = guard([E(note_index=0, directive_type="max_grid_window", hours=[], max_grid_kwh=150)], 1, 200.0, 40.0)
    assert out[0].directive_type.value == "no_op" and out[0].structured_adjustment is None


def test_missing_numeric_becomes_no_op():
    out = guard([E(note_index=0, directive_type="max_grid_window", hours=[18])], 1, 200.0, 40.0)
    assert out[0].directive_type.value == "no_op"


def test_duplicate_indices_are_realigned_positionally():
    out = guard([E(note_index=0, directive_type="no_charge_window", hours=[2]),
                 E(note_index=0, directive_type="no_discharge_window", hours=[5])], 2, 200.0, 40.0)
    assert [e.note_index for e in out] == [0, 1]
    assert {e.directive_type.value for e in out} == {"no_charge_window", "no_discharge_window"}


def test_applies_is_never_taken_from_the_model():
    out = guard([E(note_index=0, directive_type="no_op", hours=[1, 2], factor=0.5)], 1, 200.0, 40.0)
    assert out[0].applies is False and out[0].structured_adjustment is None


# --- beyond the spec's nine: hardening and edge cases ---------------------------------------


def test_flagged_reserve_given_as_percentage_is_rescaled_not_pinned_to_capacity():
    out = guard([E(note_index=0, directive_type="minimum_battery_reserve", hours=[18], minimum_energy_kwh=50,
                   reserve_is_fraction_of_capacity=True)], 1, 200.0, 40.0)
    assert out[0].structured_adjustment.model_dump()["minimum_energy_kwh"] == 100.0


def test_absolute_reserve_is_left_alone():
    out = guard([E(note_index=0, directive_type="minimum_battery_reserve", hours=[18], minimum_energy_kwh=90)], 1, 200.0, 40.0)
    assert out[0].structured_adjustment.model_dump() == {"hours": [18], "minimum_energy_kwh": 90.0}


def test_zero_values_survive_serialisation():
    solar = guard([E(note_index=0, directive_type="solar_reduction", hours=[9], factor=0.0)], 1, 200.0, 40.0)
    assert solar[0].structured_adjustment.model_dump() == {"hours": [9], "factor": 0.0}
    grid = guard([E(note_index=0, directive_type="max_grid_window", hours=[9], max_grid_kwh=0.0)], 1, 200.0, 40.0)
    assert grid[0].structured_adjustment.model_dump() == {"hours": [9], "max_grid_kwh": 0.0}
    reserve = guard([E(note_index=0, directive_type="minimum_battery_reserve", hours=[9], minimum_energy_kwh=0.0)], 1, 200.0, 40.0)
    assert reserve[0].structured_adjustment.model_dump() == {"hours": [9], "minimum_energy_kwh": 0.0}


def test_non_finite_and_negative_numbers_become_no_op():
    for kwargs in (
        dict(directive_type="solar_reduction", factor=float("nan")),
        dict(directive_type="solar_reduction", factor=float("inf")),
        dict(directive_type="max_grid_window", max_grid_kwh=float("inf")),
        dict(directive_type="max_grid_window", max_grid_kwh=-5.0),
        dict(directive_type="minimum_battery_reserve", minimum_energy_kwh=float("nan")),
    ):
        out = guard([E(note_index=0, hours=[10], **kwargs)], 1, 200.0, 40.0)
        assert out[0].directive_type.value == "no_op" and out[0].structured_adjustment is None, kwargs


def test_unknown_directive_type_becomes_no_op():
    bogus = E.model_construct(note_index=0, directive_type="grid_export", hours=[3], factor=None,
                              minimum_energy_kwh=None, reserve_is_fraction_of_capacity=False,
                              max_grid_kwh=None, explanation="")
    out = guard([bogus], 1, 200.0, 40.0)
    assert out[0].directive_type.value == "no_op" and out[0].applies is False


def test_malformed_hours_never_crash():
    def entry(hours):
        return E.model_construct(note_index=0, directive_type="no_charge_window", hours=hours, factor=None,
                                 minimum_energy_kwh=None, reserve_is_fraction_of_capacity=False,
                                 max_grid_kwh=None, explanation="")

    assert guard([entry([float("inf"), float("nan"), 13, "x", None, True, 24, -1, 10**30])], 1, 200.0, 40.0)[0] \
        .structured_adjustment.model_dump() == {"hours": [13]}
    for junk in (5, None, "1314", b"12", 3.5):
        assert guard([entry(junk)], 1, 200.0, 40.0)[0].directive_type.value == "no_op"


def test_entries_are_mapped_by_index_when_indices_are_canonical():
    out = guard([E(note_index=1, directive_type="no_discharge_window", hours=[5]),
                 E(note_index=0, directive_type="no_charge_window", hours=[2])], 2, 200.0, 40.0)
    assert [e.directive_type.value for e in out] == ["no_charge_window", "no_discharge_window"]


def test_missing_middle_entry_is_filled_with_no_op():
    out = guard([E(note_index=0, directive_type="no_charge_window", hours=[2]),
                 E(note_index=2, directive_type="no_discharge_window", hours=[5])], 3, 200.0, 40.0)
    assert [e.directive_type.value for e in out] == ["no_charge_window", "no_op", "no_discharge_window"]


def test_extra_entries_are_ignored():
    out = guard([E(note_index=0, directive_type="no_charge_window", hours=[2]),
                 E(note_index=1, directive_type="no_discharge_window", hours=[5])], 1, 200.0, 40.0)
    assert len(out) == 1 and out[0].directive_type.value == "no_charge_window"


def test_explanation_is_single_line_and_capped():
    out = guard([E(note_index=0, directive_type="no_charge_window", hours=[2],
                   explanation="line one\n\n  line   two " + "x" * 1000)], 1, 200.0, 40.0)
    text = out[0].explanation
    assert "\n" not in text and "  " not in text and len(text) <= MAX_EXPLANATION_CHARS


def test_every_output_serialises_to_the_response_shape():
    raw = [E(note_index=0, directive_type="solar_reduction", hours=[13, 14], factor=0.2),
           E(note_index=1, directive_type="no_op"),
           E(note_index=2, directive_type="max_grid_window", hours=[18], max_grid_kwh=140)]
    payload = json.loads(json.dumps([e.model_dump(mode="json") for e in guard(raw, 3, 200.0, 40.0)]))
    assert payload[0]["structured_adjustment"] == {"hours": [13, 14], "factor": 0.2}
    assert payload[1] == {"note_index": 1, "applies": False, "directive_type": "no_op",
                          "structured_adjustment": None, "explanation": payload[1]["explanation"]}
    assert payload[2]["structured_adjustment"] == {"hours": [18], "max_grid_kwh": 140.0}
    assert all(set(p) == {"note_index", "applies", "directive_type", "structured_adjustment", "explanation"} for p in payload)
