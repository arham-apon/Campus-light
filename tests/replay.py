"""Independent replay of a returned plan against the official rules."""
from __future__ import annotations

TOL = 0.01


def compile_expected(directives: list[dict], base_solar: list[float]):
    effective = list(base_solar)
    no_charge, no_discharge, reserve, grid_cap = set(), set(), {}, {}
    for entry in directives:
        if not entry.get("applies"):
            continue
        adjustment = entry.get("structured_adjustment") or {}
        hours = adjustment.get("hours", [])
        kind = entry["directive_type"]
        if kind == "solar_reduction":
            for hour in hours:
                effective[hour] = base_solar[hour] * float(adjustment["factor"])
        elif kind == "no_charge_window":
            no_charge.update(hours)
        elif kind == "no_discharge_window":
            no_discharge.update(hours)
        elif kind == "minimum_battery_reserve":
            for hour in hours:
                reserve[hour] = max(reserve.get(hour, 0.0), float(adjustment["minimum_energy_kwh"]))
        elif kind == "max_grid_window":
            for hour in hours:
                grid_cap[hour] = min(grid_cap.get(hour, float("inf")), float(adjustment["max_grid_kwh"]))
    return effective, no_charge, no_discharge, reserve, grid_cap


def replay(response: dict, request: dict, effective_solar, no_charge, no_discharge, reserve, grid_cap) -> list[str]:
    errors: list[str] = []
    hours_in = sorted(request["hours"], key=lambda h: h["hour"])
    demand = [h["demand_kwh"] for h in hours_in]
    tariff = [h["tariff_bdt_per_kwh"] for h in hours_in]
    battery = request["battery"]
    plan = response["hourly_plan"]

    if [p["hour"] for p in plan] != list(range(24)):
        return ["hourly_plan must list hours 0..23 exactly once, in order"]

    energy_prev = float(battery["initial_energy_kwh"])
    for entry in plan:
        hour = entry["hour"]
        action = entry["battery_action"]
        magnitude = float(entry["battery_kwh"])
        charge = magnitude if action == "charge" else 0.0
        discharge = magnitude if action == "discharge" else 0.0

        if action not in {"charge", "discharge", "idle"}:
            errors.append(f"h{hour}: illegal battery_action {action}")
        if action == "idle" and abs(magnitude) > 0:
            errors.append(f"h{hour}: idle with battery_kwh {magnitude}")
        if magnitude < -TOL or entry["grid_kwh"] < -TOL or entry["solar_used_kwh"] < -TOL:
            errors.append(f"h{hour}: negative value")
        if entry["solar_used_kwh"] > effective_solar[hour] + TOL:
            errors.append(f"h{hour}: solar overuse {entry['solar_used_kwh']} > {effective_solar[hour]}")
        if abs(entry["grid_kwh"] + entry["solar_used_kwh"] + discharge - demand[hour] - charge) > TOL:
            errors.append(f"h{hour}: energy balance violated")
        if abs(energy_prev + charge - discharge - entry["battery_energy_after_kwh"]) > TOL:
            errors.append(f"h{hour}: battery transition violated")
        if charge > battery["max_charge_kwh_per_hour"] + TOL:
            errors.append(f"h{hour}: charge rate exceeded")
        if discharge > battery["max_discharge_kwh_per_hour"] + TOL:
            errors.append(f"h{hour}: discharge rate exceeded")
        floor = max(float(battery["minimum_energy_kwh"]), reserve.get(hour, 0.0))
        if entry["battery_energy_after_kwh"] < floor - TOL:
            errors.append(f"h{hour}: below reserve floor {floor}")
        if entry["battery_energy_after_kwh"] > battery["capacity_kwh"] + TOL:
            errors.append(f"h{hour}: above capacity")
        if hour in no_charge and charge > TOL:
            errors.append(f"h{hour}: charged inside no_charge_window")
        if hour in no_discharge and discharge > TOL:
            errors.append(f"h{hour}: discharged inside no_discharge_window")
        if hour in grid_cap and entry["grid_kwh"] > grid_cap[hour] + TOL:
            errors.append(f"h{hour}: grid cap exceeded")
        energy_prev = entry["battery_energy_after_kwh"]

    if abs(energy_prev - float(battery["initial_energy_kwh"])) > TOL:
        errors.append("end-of-day battery neutrality violated")

    recomputed_grid = sum(p["grid_kwh"] for p in plan)
    recomputed_cost = sum(p["grid_kwh"] * tariff[p["hour"]] for p in plan)
    recomputed_peak = max(p["grid_kwh"] for p in plan)
    if abs(recomputed_grid - response["total_grid_kwh"]) > TOL:
        errors.append("total_grid_kwh disagrees with hourly_plan")
    if abs(recomputed_cost - response["total_cost_bdt"]) > TOL:
        errors.append("total_cost_bdt disagrees with hourly_plan")
    if abs(recomputed_peak - response["peak_grid_kwh"]) > TOL:
        errors.append("peak_grid_kwh disagrees with hourly_plan")
    return errors
