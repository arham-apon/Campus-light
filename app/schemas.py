"""All Pydantic contracts for the GridWise service."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

# --------------------------------------------------------------------------
# Request contract
# --------------------------------------------------------------------------


class HourInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class BatteryInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)


class OptimizeEnergyRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str = Field(min_length=1)
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput] = Field(min_length=24, max_length=24)
    battery: BatteryInput

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, v: list[str]) -> list[str]:
        if any(not isinstance(n, str) or not n.strip() for n in v):
            raise ValueError("operator_notes entries must be non-empty strings")
        return v

    @model_validator(mode="after")
    def _hours_cover_full_day(self) -> "OptimizeEnergyRequest":
        seen = [h.hour for h in self.hours]
        if sorted(seen) != list(range(24)):
            raise ValueError("hours must contain each hour 0..23 exactly once")
        self.hours = sorted(self.hours, key=lambda h: h.hour)
        return self

    # Convenience accessors used downstream.
    @property
    def demand(self) -> list[float]:
        return [float(h.demand_kwh) for h in self.hours]

    @property
    def solar(self) -> list[float]:
        return [float(h.solar_kwh) for h in self.hours]

    @property
    def tariff(self) -> list[float]:
        return [float(h.tariff_bdt_per_kwh) for h in self.hours]


# --------------------------------------------------------------------------
# Directive vocabulary
# --------------------------------------------------------------------------


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


# --------------------------------------------------------------------------
# LLM-facing extraction contract (flat on purpose)
# --------------------------------------------------------------------------


class LLMDirectiveEntry(BaseModel):
    """One model-produced interpretation, before guardrails."""

    model_config = ConfigDict(extra="ignore")

    note_index: int = Field(description="Zero-based index of the operator note this entry interprets.")
    directive_type: DirectiveType = Field(description="One supported directive type, or no_op.")
    hours: list[int] = Field(default_factory=list, description="Affected whole hours, 0-23, start inclusive, end exclusive.")
    factor: float | None = Field(default=None, description="solar_reduction only: usable solar fraction REMAINING (0-1).")
    minimum_energy_kwh: float | None = Field(default=None, description="minimum_battery_reserve only: required stored energy.")
    reserve_is_fraction_of_capacity: bool = Field(default=False, description="True when minimum_energy_kwh was stated as a share of capacity.")
    max_grid_kwh: float | None = Field(default=None, description="max_grid_window only: hourly grid import cap in kWh.")
    explanation: str = Field(default="", description="One short sentence justifying the interpretation.")


class DirectiveInterpretationPackage(BaseModel):
    """Top-level structured-output schema handed to Gemini."""

    model_config = ConfigDict(extra="ignore")

    entries: list[LLMDirectiveEntry] = Field(default_factory=list, description="Exactly one entry per operator note, in order.")


# --------------------------------------------------------------------------
# Response contract
# --------------------------------------------------------------------------


class StructuredAdjustment(BaseModel):
    """Serialises to EXACTLY the keys required by the directive type."""

    model_config = ConfigDict(extra="forbid")

    hours: list[int] | None = None
    factor: float | None = None
    minimum_energy_kwh: float | None = None
    max_grid_kwh: float | None = None

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        pairs = (
            ("hours", self.hours),
            ("factor", self.factor),
            ("minimum_energy_kwh", self.minimum_energy_kwh),
            ("max_grid_kwh", self.max_grid_kwh),
        )
        return {k: v for k, v in pairs if v is not None}


class DirectiveInterpretationEntry(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: StructuredAdjustment | None
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeEnergyResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretationEntry]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
