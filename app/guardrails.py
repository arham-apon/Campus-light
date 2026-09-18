"""Deterministic audit of untrusted model output."""
from __future__ import annotations

import math
from typing import Iterable

from app.logging_utils import get_logger
from app.schemas import (
    DirectiveInterpretationEntry,
    DirectiveType,
    LLMDirectiveEntry,
    StructuredAdjustment,
)

log = get_logger(__name__)

NO_OP_EXPLANATION = "This note does not affect today's 24-hour energy schedule."
MAX_EXPLANATION_CHARS = 300

# A model-supplied share is a fraction in [0, 1]. Values from 2 to 100 are read as percentages
# (25 -> 0.25). A value in (1, 2) is neither a sane fraction nor a sane percentage (it would mean
# 1-2 %), so it is treated as an out-of-range fraction and clamped to 1.0 (1.4 -> 1.0).
PERCENT_RESCUE_MIN = 2.0
PERCENT_RESCUE_MAX = 100.0


def _clean_explanation(text: str | None, fallback: str) -> str:
    if not text or not str(text).strip():
        return fallback
    cleaned = " ".join(str(text).split())
    return cleaned[:MAX_EXPLANATION_CHARS]


def _normalise_hours(raw: Iterable | None) -> list[int]:
    """Unique ints inside [0, 23], ascending. Anything else is dropped."""
    if not raw or isinstance(raw, (str, bytes)):
        return []
    try:
        values = list(raw)
    except TypeError:  # not iterable
        return []
    out: set[int] = set()
    for value in values:
        try:
            if isinstance(value, bool):
                continue
            hour = int(value)
        except (TypeError, ValueError, OverflowError):  # OverflowError: int(float("inf"))
            continue
        if 0 <= hour <= 23:
            out.add(hour)
    return sorted(out)


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _to_fraction(value: float) -> float:
    """Coerce a model-supplied share into [0, 1]: percentages are rescaled, anything else clamped."""
    if PERCENT_RESCUE_MIN <= value <= PERCENT_RESCUE_MAX:
        value = value / 100.0
    return min(1.0, max(0.0, value))


def _no_op(note_index: int, explanation: str = NO_OP_EXPLANATION) -> DirectiveInterpretationEntry:
    return DirectiveInterpretationEntry(
        note_index=note_index,
        applies=False,
        directive_type=DirectiveType.NO_OP,
        structured_adjustment=None,
        explanation=_clean_explanation(explanation, NO_OP_EXPLANATION),
    )


def _align_entries(raw_entries: list[LLMDirectiveEntry], num_notes: int) -> list[LLMDirectiveEntry | None]:
    """Map model entries onto note slots 0..N-1.

    Preferred path: the returned note_index values are exactly {0..N-1}.
    Fallback path: indices are wrong/duplicated, so use positional order instead.
    """
    slots: list[LLMDirectiveEntry | None] = [None] * num_notes
    indices = [e.note_index for e in raw_entries]
    if len(raw_entries) == num_notes and sorted(indices) == list(range(num_notes)):
        for entry in raw_entries:
            slots[entry.note_index] = entry
        return slots

    log.warning("note_index set %s not canonical for %d notes; falling back to positional mapping", indices, num_notes)
    used: set[int] = set()
    for entry in raw_entries:  # honour valid, unused indices first
        idx = entry.note_index
        if isinstance(idx, int) and 0 <= idx < num_notes and idx not in used and slots[idx] is None:
            slots[idx] = entry
            used.add(idx)
    leftovers = [e for e in raw_entries if e not in [s for s in slots if s is not None]]
    for position in range(num_notes):
        if slots[position] is None and leftovers:
            slots[position] = leftovers.pop(0)
    return slots


def apply_deterministic_guardrails(
    raw_entries: list[LLMDirectiveEntry],
    num_notes: int,
    battery_capacity_kwh: float,
    base_min_energy_kwh: float,
) -> list[DirectiveInterpretationEntry]:
    """Return exactly num_notes schema-perfect entries in note_index order."""
    capacity = max(0.0, float(battery_capacity_kwh))
    slots = _align_entries(list(raw_entries or []), num_notes)
    validated: list[DirectiveInterpretationEntry] = []

    for index in range(num_notes):
        entry = slots[index]
        if entry is None:
            validated.append(_no_op(index, "No valid interpretation was produced for this note; treated as non-applicable."))
            continue

        try:
            directive = DirectiveType(entry.directive_type)
        except ValueError:
            log.warning("note %d: unsupported directive_type; downgraded to no_op", index)
            validated.append(_no_op(index))
            continue

        if directive is DirectiveType.NO_OP:
            validated.append(_no_op(index, entry.explanation))
            continue

        hours = _normalise_hours(entry.hours)
        if not hours:
            log.warning("note %d: %s without usable hours; downgraded to no_op", index, directive.value)
            validated.append(_no_op(index, "No valid hour window could be extracted; treated as non-applicable."))
            continue

        adjustment: StructuredAdjustment | None = None

        if directive is DirectiveType.SOLAR_REDUCTION:
            factor = _finite(entry.factor)
            if factor is None:
                validated.append(_no_op(index, "Solar reduction was stated without a usable fraction; treated as non-applicable."))
                continue
            factor = _to_fraction(factor)  # percentage rescue + clamp to [0, 1]
            adjustment = StructuredAdjustment(hours=hours, factor=round(factor, 6))

        elif directive is DirectiveType.MINIMUM_BATTERY_RESERVE:
            reserve = _finite(entry.minimum_energy_kwh)
            if reserve is None:
                validated.append(_no_op(index, "Reserve level could not be resolved; treated as non-applicable."))
                continue
            if entry.reserve_is_fraction_of_capacity:
                reserve = _to_fraction(reserve) * capacity  # also rescues a flagged percentage (50 -> 0.5)
            elif 0.0 < reserve <= 1.0 and capacity > 1.0:
                reserve = reserve * capacity  # unflagged but clearly a share
            reserve = min(capacity, max(0.0, reserve))
            adjustment = StructuredAdjustment(hours=hours, minimum_energy_kwh=round(reserve, 6))

        elif directive is DirectiveType.MAX_GRID_WINDOW:
            cap = _finite(entry.max_grid_kwh)
            if cap is None or cap < 0.0:
                validated.append(_no_op(index, "Grid cap could not be resolved; treated as non-applicable."))
                continue
            adjustment = StructuredAdjustment(hours=hours, max_grid_kwh=round(cap, 6))

        else:  # no_charge_window / no_discharge_window
            adjustment = StructuredAdjustment(hours=hours)

        validated.append(
            DirectiveInterpretationEntry(
                note_index=index,
                applies=True,                      # R6 enforced structurally
                directive_type=directive,
                structured_adjustment=adjustment,
                explanation=_clean_explanation(entry.explanation, f"Interpreted as {directive.value}."),
            )
        )

    assert len(validated) == num_notes
    assert [e.note_index for e in validated] == list(range(num_notes))
    _ = base_min_energy_kwh  # kept in the signature: the solver applies max(base, directive)
    return validated
