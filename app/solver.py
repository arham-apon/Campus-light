"""Exact LP dispatch (SciPy HiGHS) plus deterministic plan post-processing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from app.directives import OptimizerDirectives
from app.logging_utils import get_logger
from app.schemas import BatteryInput, HourlyPlanEntry

log = get_logger(__name__)

H = 24
VARS_PER_HOUR = 5
N_VARS = H * VARS_PER_HOUR
CHURN_PENALTY = 1e-6
ACTION_EPS = 1e-6          # below this a net battery move is reported as idle
ROUND_DP = 6
TOLERANCE = 0.01


def _idx(hour: int, kind: int) -> int:
    return hour * VARS_PER_HOUR + kind


@dataclass
class DispatchResult:
    plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    relaxation: str          # "none" | "reserve" | "reserve+grid_cap" | "grid_only"
    solver_status: str


def _build_and_solve(
    demand: list[float],
    tariff: list[float],
    battery: BatteryInput,
    directives: OptimizerDirectives,
    use_reserve: bool,
    use_grid_cap: bool,
):
    capacity = float(battery.capacity_kwh)
    e_initial = float(battery.initial_energy_kwh)
    e_min = float(battery.minimum_energy_kwh)
    max_charge = float(battery.max_charge_kwh_per_hour)
    max_discharge = float(battery.max_discharge_kwh_per_hour)

    cost = np.zeros(N_VARS)
    for hour in range(H):
        cost[_idx(hour, 0)] = tariff[hour]
        cost[_idx(hour, 2)] = CHURN_PENALTY
        cost[_idx(hour, 3)] = CHURN_PENALTY

    a_eq = np.zeros((2 * H, N_VARS))
    b_eq = np.zeros(2 * H)
    for hour in range(H):
        a_eq[hour, _idx(hour, 0)] = 1.0     # grid
        a_eq[hour, _idx(hour, 1)] = 1.0     # solar used
        a_eq[hour, _idx(hour, 3)] = 1.0     # discharge
        a_eq[hour, _idx(hour, 2)] = -1.0    # charge
        b_eq[hour] = demand[hour]

        row = H + hour
        a_eq[row, _idx(hour, 4)] = 1.0
        a_eq[row, _idx(hour, 2)] = -1.0
        a_eq[row, _idx(hour, 3)] = 1.0
        if hour == 0:
            b_eq[row] = e_initial
        else:
            a_eq[row, _idx(hour - 1, 4)] = -1.0

    bounds: list[tuple[float, float | None]] = []
    for hour in range(H):
        grid_cap = directives.grid_cap_by_hour.get(hour) if use_grid_cap else None
        bounds.append((0.0, grid_cap))
        bounds.append((0.0, max(0.0, directives.effective_solar[hour])))
        bounds.append((0.0, 0.0 if hour in directives.no_charge_hours else max_charge))
        bounds.append((0.0, 0.0 if hour in directives.no_discharge_hours else max_discharge))
        if hour == H - 1:
            bounds.append((e_initial, e_initial))          # end-of-day neutrality
        else:
            floor = e_min
            if use_reserve:
                floor = max(floor, directives.reserve_by_hour.get(hour, 0.0))
            bounds.append((min(floor, capacity), capacity))

    return linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")


def _grid_only_plan(demand, directives, battery) -> list[HourlyPlanEntry]:
    """Always-feasible last resort: battery idle all day, free solar consumed first."""
    plan = []
    energy = float(battery.initial_energy_kwh)
    for hour in range(H):
        solar_used = round(min(demand[hour], max(0.0, directives.effective_solar[hour])), ROUND_DP)
        grid = round(max(0.0, demand[hour] - solar_used), ROUND_DP)
        plan.append(
            HourlyPlanEntry(
                hour=hour,
                grid_kwh=grid,
                solar_used_kwh=solar_used,
                battery_action="idle",
                battery_kwh=0.0,
                battery_energy_after_kwh=round(energy, ROUND_DP),
            )
        )
    return plan


def _build_plan(x: np.ndarray, demand: list[float], directives: OptimizerDirectives, battery: BatteryInput) -> list[HourlyPlanEntry]:
    """Net charge/discharge, round, and rebuild state so the judge's replay is exact."""
    e_initial = float(battery.initial_energy_kwh)
    plan: list[HourlyPlanEntry] = []
    energy_prev = e_initial

    for hour in range(H):
        solar_used = min(max(x[_idx(hour, 1)], 0.0), max(0.0, directives.effective_solar[hour]))
        solar_used = round(solar_used, ROUND_DP)

        net = x[_idx(hour, 2)] - x[_idx(hour, 3)]
        if abs(net) < ACTION_EPS:
            net = 0.0
        net = round(net, ROUND_DP)
        if hour == H - 1:                       # snap away any float drift
            net = round(e_initial - energy_prev, ROUND_DP)

        energy = round(energy_prev + net, ROUND_DP)
        charge = max(net, 0.0)
        discharge = max(-net, 0.0)

        grid = round(demand[hour] + charge - discharge - solar_used, ROUND_DP)
        if grid < 0.0:                          # rounding can only ever push it a hair negative
            solar_used = round(solar_used + grid, ROUND_DP)
            grid = 0.0

        action = "idle" if net == 0.0 else ("charge" if net > 0.0 else "discharge")
        plan.append(
            HourlyPlanEntry(
                hour=hour,
                grid_kwh=grid,
                solar_used_kwh=solar_used,
                battery_action=action,
                battery_kwh=round(abs(net), ROUND_DP),
                battery_energy_after_kwh=energy,
            )
        )
        energy_prev = energy

    return plan


def solve_energy_dispatch(
    demand: list[float],
    tariff: list[float],
    battery: BatteryInput,
    directives: OptimizerDirectives,
) -> DispatchResult:
    """Solve the dispatch LP, degrading through a fixed relaxation ladder if infeasible."""
    ladder = [
        ("none", True, True),
        ("reserve", False, True),
        ("reserve+grid_cap", False, False),
    ]

    for label, use_reserve, use_grid_cap in ladder:
        result = _build_and_solve(demand, tariff, battery, directives, use_reserve, use_grid_cap)
        if result.success:
            if label != "none":
                log.warning("LP infeasible with full directive set; relaxed=%s", label)
            plan = _build_plan(result.x, demand, directives, battery)
            return _finalise(plan, tariff, label, str(result.message))
        log.warning("LP attempt '%s' failed: %s", label, result.message)

    log.error("LP infeasible at every relaxation level; emitting grid-only fallback plan")
    plan = _grid_only_plan(demand, directives, battery)
    return _finalise(plan, tariff, "grid_only", "fallback")


def _finalise(plan: list[HourlyPlanEntry], tariff: list[float], relaxation: str, status: str) -> DispatchResult:
    total_grid = round(sum(p.grid_kwh for p in plan), ROUND_DP)
    total_cost = round(sum(p.grid_kwh * tariff[p.hour] for p in plan), ROUND_DP)
    peak_grid = round(max(p.grid_kwh for p in plan), ROUND_DP)
    return DispatchResult(
        plan=plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        relaxation=relaxation,
        solver_status=status,
    )


def self_check(plan: list[HourlyPlanEntry], demand, directives, battery) -> list[str]:
    """Cheap internal replay; findings are logged, never returned to the caller."""
    problems: list[str] = []
    energy_prev = float(battery.initial_energy_kwh)
    for entry in plan:
        hour = entry.hour
        charge = entry.battery_kwh if entry.battery_action == "charge" else 0.0
        discharge = entry.battery_kwh if entry.battery_action == "discharge" else 0.0
        if abs(entry.grid_kwh + entry.solar_used_kwh + discharge - demand[hour] - charge) > TOLERANCE:
            problems.append(f"h{hour}:balance")
        if abs(energy_prev + charge - discharge - entry.battery_energy_after_kwh) > TOLERANCE:
            problems.append(f"h{hour}:transition")
        if entry.solar_used_kwh > directives.effective_solar[hour] + TOLERANCE:
            problems.append(f"h{hour}:solar")
        energy_prev = entry.battery_energy_after_kwh
    if abs(energy_prev - float(battery.initial_energy_kwh)) > TOLERANCE:
        problems.append("neutrality")
    return problems
