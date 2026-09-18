"""GridWise service: intake -> LLM -> guardrails -> LP -> response."""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import llm_client
from app.config import get_settings
from app.directives import compile_directives
from app.guardrails import apply_deterministic_guardrails
from app.logging_utils import configure_logging, get_logger
from app.schemas import OptimizeEnergyRequest, OptimizeEnergyResponse
from app.solver import solve_energy_dispatch, self_check
from app.summary import build_plan_summary

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("gridwise")

# The LLM client bounds itself with llm_total_budget_seconds; this outer guard only has to be a
# little longer than that so it never cuts a hedged/fallback request short, while the whole request
# still finishes inside the 30 s hard limit.
LLM_HARD_TIMEOUT_SECONDS = settings.llm_total_budget_seconds + 2.0


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the SDK import, client and TLS connection in the background: startup and /health must
    # never wait on (or depend on) the network.
    llm_client.warmup_async()
    log.info("gridwise ready models=%s llm_enabled=%s", settings.model_chain, settings.llm_enabled)
    yield


app = FastAPI(
    title="GridWise Energy Optimization Service",
    version="1.0.0",
    description="LLM-assisted operator-directive interpretation with exact LP energy dispatch.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Error handling: structural problems -> 400, everything unexpected -> clean 500
# --------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = []
    for error in exc.errors()[:8]:
        location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
        fields.append({"field": location or "body", "issue": error.get("msg", "invalid")})
    log.warning("rejected malformed request: %s", fields)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "invalid_request", "detail": "Request payload does not match the required schema.", "fields": fields},
    )


@app.exception_handler(Exception)
async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    incident = uuid.uuid4().hex[:12]
    log.exception("unhandled error incident=%s type=%s", incident, type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": "The service could not complete this request.", "incident_id": incident},
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Never touches the LLM or the solver, so readiness is immediate."""
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeEnergyResponse, response_model_exclude_none=False)
async def optimize_energy(payload: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    started = time.perf_counter()
    notes = [note.strip() for note in payload.operator_notes]

    # 1. LLM interpretation (mandatory path), hard-bounded in time.
    try:
        extraction = await asyncio.wait_for(
            asyncio.to_thread(llm_client.extract_directives, notes, float(payload.battery.capacity_kwh)),
            timeout=LLM_HARD_TIMEOUT_SECONDS,
        )
        raw_entries, source = extraction.entries, f"{extraction.source}:{extraction.model or '-'}"
    except asyncio.TimeoutError:
        log.error("llm extraction hard timeout; degrading to no_op interpretations")
        raw_entries, source = [], "timeout"

    # 2. Deterministic guardrails.
    interpretation = apply_deterministic_guardrails(
        raw_entries,
        num_notes=len(notes),
        battery_capacity_kwh=float(payload.battery.capacity_kwh),
        base_min_energy_kwh=float(payload.battery.minimum_energy_kwh),
    )

    # 3. Compile directives and solve the dispatch LP.
    directives = compile_directives(interpretation, payload.solar)
    dispatch = solve_energy_dispatch(payload.demand, payload.tariff, payload.battery, directives)

    # 4. Internal replay; log only, response stays clean.
    problems = self_check(dispatch.plan, payload.demand, directives, payload.battery)
    if problems:
        log.error("self-check findings scenario=%s %s", payload.scenario_id, problems[:6])

    summary = build_plan_summary(interpretation, dispatch.plan, directives, dispatch.total_cost_bdt, dispatch.peak_grid_kwh)

    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info(
        "scenario=%s notes=%d llm=%s relaxation=%s cost=%.2f peak=%.2f %.0fms",
        payload.scenario_id, len(notes), source, dispatch.relaxation,
        dispatch.total_cost_bdt, dispatch.peak_grid_kwh, elapsed_ms,
    )

    return OptimizeEnergyResponse(
        scenario_id=payload.scenario_id,
        directive_interpretation=interpretation,
        hourly_plan=dispatch.plan,
        total_grid_kwh=dispatch.total_grid_kwh,
        total_cost_bdt=dispatch.total_cost_bdt,
        peak_grid_kwh=dispatch.peak_grid_kwh,
        plan_summary=summary,
    )
