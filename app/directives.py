"""Compile validated directives into optimizer inputs."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import DirectiveInterpretationEntry, DirectiveType


@dataclass
class OptimizerDirectives:
    effective_solar: list[float]
    no_charge_hours: set[int] = field(default_factory=set)
    no_discharge_hours: set[int] = field(default_factory=set)
    reserve_by_hour: dict[int, float] = field(default_factory=dict)
    grid_cap_by_hour: dict[int, float] = field(default_factory=dict)
    solar_reduced_hours: set[int] = field(default_factory=set)


def compile_directives(
    entries: list[DirectiveInterpretationEntry],
    base_solar: list[float],
) -> OptimizerDirectives:
    compiled = OptimizerDirectives(effective_solar=[float(v) for v in base_solar])

    for entry in entries:
        if not entry.applies or entry.structured_adjustment is None:
            continue
        adjustment = entry.structured_adjustment
        hours = adjustment.hours or []

        if entry.directive_type is DirectiveType.SOLAR_REDUCTION:
            factor = float(adjustment.factor or 0.0)
            for hour in hours:
                compiled.effective_solar[hour] *= factor
                compiled.solar_reduced_hours.add(hour)

        elif entry.directive_type is DirectiveType.NO_CHARGE_WINDOW:
            compiled.no_charge_hours.update(hours)

        elif entry.directive_type is DirectiveType.NO_DISCHARGE_WINDOW:
            compiled.no_discharge_hours.update(hours)

        elif entry.directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
            level = float(adjustment.minimum_energy_kwh or 0.0)
            for hour in hours:
                compiled.reserve_by_hour[hour] = max(compiled.reserve_by_hour.get(hour, 0.0), level)

        elif entry.directive_type is DirectiveType.MAX_GRID_WINDOW:
            cap = float(adjustment.max_grid_kwh or 0.0)
            for hour in hours:
                compiled.grid_cap_by_hour[hour] = min(compiled.grid_cap_by_hour.get(hour, float("inf")), cap)

    compiled.effective_solar = [max(0.0, v) for v in compiled.effective_solar]
    return compiled
