from __future__ import annotations

from app.directives import OptimizerDirectives
from app.schemas import BatteryInput
from app.solver import solve_energy_dispatch

BATTERY = BatteryInput(capacity_kwh=200, initial_energy_kwh=100, minimum_energy_kwh=40,
                       max_charge_kwh_per_hour=50, max_discharge_kwh_per_hour=50)


def _flat(demand=150.0, solar=0.0):
    return [demand] * 24, [solar] * 24


def test_arbitrage_uses_cheap_hours():
    demand, solar = _flat()
    tariff = [5.0] * 12 + [20.0] * 12
    result = solve_energy_dispatch(demand, tariff, BATTERY, OptimizerDirectives(effective_solar=solar))
    assert result.relaxation == "none"
    assert any(p.battery_action == "charge" for p in result.plan)
    assert any(p.battery_action == "discharge" for p in result.plan)
    assert result.plan[23].battery_energy_after_kwh == BATTERY.initial_energy_kwh


def test_no_charge_window_is_respected():
    demand, solar = _flat()
    tariff = [5.0] * 12 + [20.0] * 12
    directives = OptimizerDirectives(effective_solar=solar, no_charge_hours={0, 1, 2, 3, 4, 5})
    result = solve_energy_dispatch(demand, tariff, BATTERY, directives)
    assert all(result.plan[h].battery_action != "charge" for h in range(6))


def test_grid_cap_is_respected():
    demand, solar = _flat()
    tariff = [10.0] * 24
    directives = OptimizerDirectives(effective_solar=solar, grid_cap_by_hour={h: 120.0 for h in (18, 19, 20)})
    result = solve_energy_dispatch(demand, tariff, BATTERY, directives)
    assert all(result.plan[h].grid_kwh <= 120.0 + 0.01 for h in (18, 19, 20))


def test_reserve_floor_is_respected():
    demand, solar = _flat()
    tariff = [5.0] * 12 + [20.0] * 12
    directives = OptimizerDirectives(effective_solar=solar, reserve_by_hour={h: 150.0 for h in (18, 19, 20)})
    result = solve_energy_dispatch(demand, tariff, BATTERY, directives)
    assert all(result.plan[h].battery_energy_after_kwh >= 150.0 - 0.01 for h in (18, 19, 20))


def test_solar_is_consumed_before_grid():
    demand, _ = _flat()
    solar = [0.0] * 8 + [120.0] * 8 + [0.0] * 8
    tariff = [10.0] * 24
    result = solve_energy_dispatch(demand, tariff, BATTERY, OptimizerDirectives(effective_solar=solar))
    assert all(abs(result.plan[h].solar_used_kwh - 120.0) < 0.01 for h in range(8, 16))


def test_infeasible_directives_degrade_instead_of_crashing():
    demand, solar = _flat()
    tariff = [10.0] * 24
    # A 30 kWh hourly cap cannot meet 150 kWh demand with no solar: the ladder must still return a plan.
    directives = OptimizerDirectives(effective_solar=solar, grid_cap_by_hour={h: 30.0 for h in range(24)})
    result = solve_energy_dispatch(demand, tariff, BATTERY, directives)
    assert len(result.plan) == 24
    assert result.relaxation in {"reserve+grid_cap", "grid_only"}


def test_idle_hours_carry_zero_magnitude():
    demand, solar = _flat()
    result = solve_energy_dispatch(demand, [10.0] * 24, BATTERY, OptimizerDirectives(effective_solar=solar))
    assert all(p.battery_kwh == 0.0 for p in result.plan if p.battery_action == "idle")
