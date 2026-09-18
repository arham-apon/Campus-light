"""Deterministic, zero-latency plan_summary text."""
from __future__ import annotations

from app.directives import OptimizerDirectives
from app.schemas import DirectiveInterpretationEntry, DirectiveType, HourlyPlanEntry

_LABELS = {
    DirectiveType.SOLAR_REDUCTION: "reduced solar availability",
    DirectiveType.MINIMUM_BATTERY_RESERVE: "a raised battery reserve",
    DirectiveType.NO_CHARGE_WINDOW: "a charging blackout window",
    DirectiveType.NO_DISCHARGE_WINDOW: "a discharging blackout window",
    DirectiveType.MAX_GRID_WINDOW: "an hourly grid import cap",
}


def build_plan_summary(
    entries: list[DirectiveInterpretationEntry],
    plan: list[HourlyPlanEntry],
    directives: OptimizerDirectives,
    total_cost: float,
    peak_grid: float,
) -> str:
    applied = [_LABELS[e.directive_type] for e in entries if e.applies and e.directive_type in _LABELS]
    ignored = sum(1 for e in entries if not e.applies)

    charge_hours = [p.hour for p in plan if p.battery_action == "charge"]
    discharge_hours = [p.hour for p in plan if p.battery_action == "discharge"]

    if applied:
        head = "Applied " + ", ".join(sorted(set(applied))) + "."
    else:
        head = "No operator note changed the scheduling model."
    if ignored:
        head += f" {ignored} note(s) were non-operational and treated as no_op."

    body = (
        f" The plan buys grid energy in the cheapest feasible hours, charges the battery in "
        f"{len(charge_hours)} hour(s) and discharges it in {len(discharge_hours)} hour(s) to cover "
        f"expensive periods, uses available solar before grid import, and returns the battery to its "
        f"starting state of charge by hour 23."
    )
    tail = f" Total grid cost is {total_cost:.2f} BDT with a peak hourly import of {peak_grid:.2f} kWh."
    return (head + body + tail).strip()
