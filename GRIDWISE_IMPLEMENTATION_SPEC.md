# GridWise Energy Optimization Challenge — Turnkey Implementation Specification

**Target executor:** an autonomous AI coding agent (Claude Code), working without human intervention.
**Deliverable:** a competition-grade FastAPI microservice implementing LLM-assisted operator-directive interpretation plus exact LP energy dispatch, fully tested, containerized, and documented.
**Event:** BUP CSE Fest 2026 Hackathon — Online Preliminary Round (4 hours, 7:00 PM – 11:00 PM).

---

## 0. How to execute this document

Work through Phases 1 → 9 **in order**. Every phase is closed: it names the files to create, gives the complete file contents, and ends with a verification command that must pass before moving on. Do not improvise APIs, field names, or formulas — everything judged is fixed by the canonical Problem Statement and is reproduced verbatim in §0.2 below.

### 0.1 Hard rules that override any instinct to "improve" things

| # | Rule |
|---|---|
| R1 | Endpoint names are exactly `GET /health` and `POST /optimize-energy`. No prefixes, no `/api/v1`. |
| R2 | A language model must produce the structured interpretation of `operator_notes` that actually feeds the optimizer. Using an LLM only for `plan_summary` **fails the mandatory requirement and disqualifies the submission**. |
| R3 | LLM output is untrusted. It passes through deterministic guardrails before touching any math. |
| R4 | Time windows are whole-hour, **start-inclusive / end-exclusive**: "1 PM to 3 PM" → `[13, 14]`. |
| R5 | `solar_reduction.factor` is the **fraction that remains usable**: "80% reduction" → `0.2`; "reduced to 25%" → `0.25`. |
| R6 | `applies` is `true` for every non-`no_op` directive and `false` **only** for `no_op`, where `structured_adjustment` must be `null`. |
| R7 | `hours` arrays contain unique integers in `[0, 23]`, ascending. |
| R8 | `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` must be recomputed **from the returned `hourly_plan`**, which is the source of truth. |
| R9 | Final `battery_energy_after_kwh` at hour 23 must equal `battery.initial_energy_kwh`. |
| R10 | No secrets in the repo, the image, the logs, or any HTTP response. No raw stack traces in responses. |
| R11 | Numeric tolerance is 0.01 kWh / 0.01 BDT absolute. |
| R12 | `POST /optimize-energy` must answer within 30 s; target p95 ≤ 5 s. `GET /health` must be ready within 60 s of start and must never depend on the LLM. |

### 0.2 Canonical reference tables (copy into code as the single source of truth)

**Supported directive types and their required `structured_adjustment` shape**

| `directive_type` | Meaning | Required `structured_adjustment` | Optimizer effect |
|---|---|---|---|
| `solar_reduction` | Usable solar drops in listed hours | `{"hours": [...], "factor": number}` | `effective_solar[h] = solar[h] * factor` |
| `minimum_battery_reserve` | Battery must hold at least a level | `{"hours": [...], "minimum_energy_kwh": number}` | `E[h] >= max(base_min, directive_min)` |
| `no_charge_window` | Charging unavailable | `{"hours": [...]}` | `charge[h] = 0` |
| `no_discharge_window` | Discharging unavailable | `{"hours": [...]}` | `discharge[h] = 0` |
| `max_grid_window` | Grid import capped | `{"hours": [...], "max_grid_kwh": number}` | `grid[h] <= max_grid_kwh` |
| `no_op` | Irrelevant / distractor note | `null` | none |

**Energy rules (judge replays these hour by hour)**

```
energy balance      grid_kwh[h] + solar_used_kwh[h] + discharge[h] = demand_kwh[h] + charge[h]
battery transition  E[h] = E[h-1] + charge[h] - discharge[h],      E[-1] = initial_energy_kwh
battery bounds      max(minimum_energy_kwh, reserve[h]) <= E[h] <= capacity_kwh
rate limits         charge[h] <= max_charge_kwh_per_hour ; discharge[h] <= max_discharge_kwh_per_hour
solar               0 <= solar_used_kwh[h] <= effective_solar[h]        (surplus is curtailed, no export)
neutrality          E[23] = initial_energy_kwh
objective           minimise SUM_h grid_kwh[h] * tariff_bdt_per_kwh[h]
action consistency  battery_action ∈ {charge, discharge, idle}; battery_kwh >= 0; battery_kwh = 0 when idle
```

**Scoring map (100 pts) — what each phase buys**

| Category | Pts | Phases that earn it |
|---|---|---|
| LLM Directive Interpretation | 25 | 3, 4 |
| Directive Application & Constraint Correctness | 25 | 4, 5 |
| Optimization Quality | 10 | 5 |
| API Contract & Schema | 10 | 2, 6 |
| Performance & Reliability | 10 | 3, 6 |
| Deployment & Docker Fallback | 10 | 8 |
| Documentation & Local Reproducibility | 10 | 9 |

**Verified public-sample benchmarks** — the LP model in Phase 5 reproduces every one of these *exactly* (verified before this document was written; treat any deviation as a bug in your implementation, not in the table).

| Case | Label | `total_grid_kwh` | `total_cost_bdt` | `peak_grid_kwh` |
|---|---|---|---|---|
| SAMPLE-01 | Solar cleaning + distractor | 2692.5 | 38365 | 175 |
| SAMPLE-02 | Battery charging maintenance | 2915 | 42885 | 180 |
| SAMPLE-03 | Emergency reserve as percentage | 2430 | 35480 | 205 |
| SAMPLE-04 | No-discharge protection test | 2645 | 40495 | 225 |
| SAMPLE-05 | Temporary feeder grid cap | 2430 | 33950 | 175 |
| SAMPLE-06 | Multiple notes with distractor | 2395 | 34090 | 175 |
| SAMPLE-07 | Reserve plus transformer cap | 2560 | 38550 | 185 |
| SAMPLE-08 | Separate charge/discharge outages | 2490 | 37665 | 210 |
| SAMPLE-09 | Reduction wording normalization | 2504 | 34873 | 170 |
| SAMPLE-10 | Multi-constraint evening operation | 2715 | 41620 | 190 |

---

## Phase 1 — Environment setup & project layout

### 1.1 Directory tree

Create exactly this structure:

```
gridwise/
├── app/
│   ├── __init__.py
│   ├── config.py            # env-driven settings, no secrets in code
│   ├── schemas.py           # all Pydantic contracts (request, LLM, response)
│   ├── prompts.py           # system instruction + note rendering
│   ├── llm_client.py        # Gemini 2.0 Flash structured extraction + caching
│   ├── guardrails.py        # deterministic validation / normalisation layer
│   ├── directives.py        # compiles validated directives into optimizer inputs
│   ├── solver.py            # SciPy HiGHS LP + plan post-processing
│   ├── summary.py           # deterministic plan_summary text
│   ├── logging_utils.py     # redacting logger
│   └── main.py              # FastAPI app, routes, error handlers
├── tests/
│   ├── __init__.py
│   ├── replay.py            # independent judge-style validator
│   ├── conftest.py
│   ├── test_api_contract.py
│   ├── test_guardrails.py
│   ├── test_solver.py
│   ├── test_samples.py      # all 10 public cases, end-to-end
│   └── test_llm_live.py     # opt-in real-Gemini extraction accuracy
├── data/
│   └── BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── Makefile
└── README.md
```

Copy the provided `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json` into `data/` unmodified.

### 1.2 `requirements.txt`

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
google-genai>=1.0.0,<2.0.0
scipy==1.14.1
numpy==2.1.3
httpx==0.28.1
python-dotenv==1.0.1
pytest==8.3.4
```

`google-genai` is range-pinned because the SDK ships frequently; immediately after the first successful install run:

```bash
pip freeze > requirements.lock.txt
```

and commit the lock file. The README must tell judges that `requirements.txt` is the install surface and `requirements.lock.txt` is the byte-exact environment.

### 1.3 Bootstrap commands

```bash
mkdir -p gridwise/app gridwise/tests gridwise/data && cd gridwise
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements.lock.txt
```

### 1.4 `.env.example`

```text
# Copy to .env and fill in locally. NEVER commit .env.
GEMINI_API_KEY=your_google_ai_studio_key_here

# Optional overrides (defaults shown)
GEMINI_MODEL=gemini-2.0-flash
LLM_TIMEOUT_SECONDS=8
LLM_MAX_RETRIES=1
LLM_ENABLED=true
LLM_CACHE_SIZE=512
LOG_LEVEL=INFO
PORT=8000
```

### 1.5 `.gitignore`

```text
.env
.env.*
!.env.example
*.key
*.pem
secrets/
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
.DS_Store
.idea/
.vscode/
*.log
```

### 1.6 `.dockerignore`

```text
.git
.gitignore
.venv
venv
__pycache__
*.pyc
.pytest_cache
.env
.env.*
tests
README.md
Makefile
*.log
```

### 1.7 `Makefile` (convenience, also quoted in the README)

```makefile
.PHONY: install run test docker-build docker-run tunnel

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-build:
	docker build -t gridwise:latest .

docker-run:
	docker run --rm -p 8000:8000 -e GEMINI_API_KEY=$$GEMINI_API_KEY gridwise:latest

tunnel:
	cloudflared tunnel --no-autoupdate --url http://localhost:8000
```

**Verify Phase 1:** `python -c "import fastapi, scipy, numpy, google.genai; print('deps ok')"`

---

## Phase 2 — Pydantic data contracts

Create `app/schemas.py` with the complete file below. Three families live here: the **request** contract (what judges send), the **LLM-facing** contract (deliberately flat — see note), and the **response** contract (what judges score).

> **Design note — why the LLM schema is flat.** Structured-output schemas with nested optional objects and unions are the main source of extraction failures. The model is therefore asked for a flat entry with nullable scalars, and the strict nested `structured_adjustment` is *built deterministically* by the guardrail layer. The model is also **not** asked for `applies`: that field is derived (`applies = directive_type != no_op`), which makes rule R6 structurally impossible to violate.

```python
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
```

Also create `app/__init__.py` (empty) and `app/config.py`:

```python
"""Environment-driven settings. No secret ever has a default value."""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=False)


class Settings(BaseModel):
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    llm_timeout_seconds: float = 8.0
    llm_max_retries: int = 1
    llm_enabled: bool = True
    llm_cache_size: int = 512
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    def _f(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except ValueError:
            return default

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        llm_timeout_seconds=_f("LLM_TIMEOUT_SECONDS", 8.0),
        llm_max_retries=int(_f("LLM_MAX_RETRIES", 1)),
        llm_enabled=os.getenv("LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes"},
        llm_cache_size=int(_f("LLM_CACHE_SIZE", 512)),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
```

And `app/logging_utils.py`:

```python
"""Logger that can never print a credential."""
from __future__ import annotations

import logging
import os
import re
import sys

_REDACTIONS: list[re.Pattern[str]] = [
    re.compile(r"AIza[0-9A-Za-z\-_]{10,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*\S+"),
]


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        key = os.getenv("GEMINI_API_KEY")
        if key:
            msg = msg.replace(key, "***redacted***")
        for pattern in _REDACTIONS:
            msg = pattern.sub("***redacted***", msg)
        return msg


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(name)s :: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

**Verify Phase 2:**

```bash
python - <<'PY'
from app.schemas import OptimizeEnergyRequest, StructuredAdjustment
sa = StructuredAdjustment(hours=[13,14], factor=0.2)
assert sa.model_dump() == {"hours":[13,14],"factor":0.2}, sa.model_dump()
print("schemas ok")
PY
```

---

## Phase 3 — Gemini 2.0 Flash extraction pipeline

### 3.1 `app/prompts.py`

The system instruction carries all domain rules. Keep the examples *paraphrased away* from the public sample wording — hidden notes are deliberately reworded, and over-fitting to public phrasing is explicitly penalised.

```python
"""Prompt assets for operator-note interpretation."""
from __future__ import annotations

SYSTEM_INSTRUCTION = """\
You convert short campus-operator notes into structured energy-scheduling directives for a
24-hour optimisation model. You return JSON only, matching the provided schema exactly.

OUTPUT RULE
Return exactly one entry per operator note, in the same order, with note_index 0, 1, ... N-1.
Never merge two notes into one entry. Never split one note into two entries.

SUPPORTED DIRECTIVE TYPES (no others exist)
1. solar_reduction          - usable rooftop solar is reduced during specific hours.
                              fields: hours, factor
2. minimum_battery_reserve  - the battery must hold at least some energy during specific hours.
                              fields: hours, minimum_energy_kwh (+ reserve_is_fraction_of_capacity)
3. no_charge_window         - the battery cannot be charged during specific hours.
                              fields: hours
4. no_discharge_window      - the battery cannot be discharged during specific hours.
                              fields: hours
5. max_grid_window          - hourly grid import is capped during specific hours.
                              fields: hours, max_grid_kwh
6. no_op                    - the note does not change today's 24-hour energy schedule.
                              fields: none (leave hours empty and all numbers null)

TIME WINDOWS
- Hours are whole integers 0-23 in 24-hour clock form. noon = 12, midnight = 0.
- A range is START-INCLUSIVE and END-EXCLUSIVE.
    "from 1 PM to 3 PM"        -> [13, 14]
    "between 11 AM and 2 PM"   -> [11, 12, 13]
    "from noon until 2 PM"     -> [12, 13]
    "2 AM until 5 AM"          -> [2, 3, 4]
- A single stated hour such as "during the 5 PM hour" -> [17].
- "from 6 PM onwards" / "for the rest of the evening" -> [18, 19, 20, 21, 22, 23].
- "all day" / "throughout the day" -> 0 through 23.
- A window crossing midnight, e.g. "10 PM until 2 AM" -> [22, 23, 0, 1].
- Return hours sorted ascending with no duplicates.

SOLAR FACTOR (this is the most common mistake - read twice)
factor is the fraction of forecast solar that REMAINS USABLE, not the size of the loss.
    "output will drop to about 20%"          -> factor 0.20
    "expect an 80% reduction"                -> factor 0.20
    "roughly one fifth of normal output"     -> factor 0.20
    "about half the forecast"                -> factor 0.50
    "treated as roughly 25% of forecast"     -> factor 0.25
    "panels fully covered / no solar"        -> factor 0.0
factor must be between 0 and 1 inclusive.

BATTERY RESERVE
- If the note gives an absolute energy, e.g. "keep at least 90 kWh", set
  minimum_energy_kwh = 90 and reserve_is_fraction_of_capacity = false.
- If the note gives a share of capacity, e.g. "keep at least 50% of capacity" or
  "keep the battery at least half full", set minimum_energy_kwh to that share as a decimal
  (0.5) and reserve_is_fraction_of_capacity = true. Deterministic code multiplies by capacity.

GRID CAP
max_grid_kwh is the maximum kWh that may be imported in ANY SINGLE hour of the window,
not a total for the window.

DISTRACTORS
Many notes are realistic campus announcements with no energy effect: cafeteria menus,
library or book-return hours, sports or club registrations, seminar room bookings, exam
notices, staff meetings, parking changes. These are no_op. Do not invent an energy rule for
them and do not guess an hour range from them.

STRICT LIMITS
- Never invent demand, solar, tariff, or battery values.
- Never emit a directive type outside the six listed above.
- If a note is genuinely ambiguous or unsupported, choose no_op rather than guessing.
- explanation is one short factual sentence. It is never scored word for word.
"""

FEW_SHOT = """\
Worked examples (different wording from the notes you will receive):

Notes:
0. "Inverter servicing will cut rooftop generation to roughly one third between 10:00 and 13:00."
1. "Payroll forms are due at the admin office on Sunday."
Entries:
0 -> solar_reduction, hours [10, 11, 12], factor 0.33
1 -> no_op

Notes:
0. "Please avoid drawing more than 140 kWh per hour from the grid between 6 and 8 in the evening."
Entries:
0 -> max_grid_window, hours [18, 19], max_grid_kwh 140

Notes:
0. "The storage bank should stay at least three quarters full through the late evening, 9 PM to 11 PM."
Entries:
0 -> minimum_battery_reserve, hours [21, 22], minimum_energy_kwh 0.75,
     reserve_is_fraction_of_capacity true

Notes:
0. "Storage must not feed the campus while relay tests run from 4 PM to 6 PM."
1. "Charger cabinet is isolated for inspection, 11 AM to 1 PM."
Entries:
0 -> no_discharge_window, hours [16, 17]
1 -> no_charge_window, hours [11, 12]
"""


def build_user_prompt(notes: list[str], battery_capacity_kwh: float) -> str:
    numbered = "\n".join(f"{i}. {note.strip()}" for i, note in enumerate(notes))
    return (
        f"{FEW_SHOT}\n"
        f"Battery capacity for this scenario: {battery_capacity_kwh:g} kWh.\n"
        f"Interpret the following {len(notes)} operator note(s) and return exactly "
        f"{len(notes)} entries with note_index 0..{len(notes) - 1}.\n\n"
        f"OPERATOR NOTES:\n{numbered}\n"
    )
```

### 3.2 `app/llm_client.py`

```python
"""Gemini 2.0 Flash structured extraction with timeout, retry, cache and safe degradation."""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any

from app.config import get_settings
from app.logging_utils import get_logger
from app.prompts import SYSTEM_INSTRUCTION, build_user_prompt
from app.schemas import DirectiveInterpretationPackage, LLMDirectiveEntry

log = get_logger(__name__)

_client: Any = None
_client_lock = threading.Lock()
_cache: "OrderedDict[tuple, list[LLMDirectiveEntry]]" = OrderedDict()
_cache_lock = threading.Lock()


class ExtractionResult:
    """Carries the entries plus provenance for logging and diagnostics."""

    __slots__ = ("entries", "source", "latency_ms", "error")

    def __init__(self, entries: list[LLMDirectiveEntry], source: str, latency_ms: float, error: str | None = None):
        self.entries = entries
        self.source = source
        self.latency_ms = latency_ms
        self.error = error


def _get_client():
    """Lazily build one shared client. Returns None when no key is configured."""
    global _client
    if _client is not None:
        return _client
    settings = get_settings()
    if not settings.gemini_api_key or not settings.llm_enabled:
        return None
    with _client_lock:
        if _client is None:
            from google import genai
            from google.genai import types

            timeout_ms = int(settings.llm_timeout_seconds * 1000)
            try:
                _client = genai.Client(
                    api_key=settings.gemini_api_key,
                    http_options=types.HttpOptions(timeout=timeout_ms),
                )
            except TypeError:  # older SDK without http_options
                _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def warmup() -> None:
    """Build the client at startup so the first judged request pays no import cost."""
    try:
        _get_client()
    except Exception as exc:  # never block startup
        log.warning("llm warmup skipped: %s", type(exc).__name__)


def _cache_get(key: tuple) -> list[LLMDirectiveEntry] | None:
    with _cache_lock:
        entries = _cache.get(key)
        if entries is not None:
            _cache.move_to_end(key)
        return entries


def _cache_put(key: tuple, entries: list[LLMDirectiveEntry]) -> None:
    limit = get_settings().llm_cache_size
    if limit <= 0:
        return
    with _cache_lock:
        _cache[key] = entries
        _cache.move_to_end(key)
        while len(_cache) > limit:
            _cache.popitem(last=False)


def _parse_response(response: Any) -> list[LLMDirectiveEntry]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, DirectiveInterpretationPackage):
        return list(parsed.entries)
    if isinstance(parsed, dict):
        return list(DirectiveInterpretationPackage.model_validate(parsed).entries)

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ValueError("empty model response")
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model response")
    payload = json.loads(text[start : end + 1])
    return list(DirectiveInterpretationPackage.model_validate(payload).entries)


def extract_directives(notes: list[str], battery_capacity_kwh: float) -> ExtractionResult:
    """Interpret operator notes with Gemini. Never raises; degrades to an empty list."""
    started = time.perf_counter()
    key = (tuple(n.strip() for n in notes), round(float(battery_capacity_kwh), 6))

    cached = _cache_get(key)
    if cached is not None:
        return ExtractionResult(list(cached), "cache", (time.perf_counter() - started) * 1000)

    client = _get_client()
    if client is None:
        log.error("LLM unavailable: GEMINI_API_KEY missing or LLM_ENABLED=false")
        return ExtractionResult([], "unavailable", (time.perf_counter() - started) * 1000, "llm_unavailable")

    from google.genai import types

    settings = get_settings()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
        top_p=1.0,
        candidate_count=1,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=DirectiveInterpretationPackage,
    )
    prompt = build_user_prompt(notes, battery_capacity_kwh)

    last_error: str | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=config,
            )
            entries = _parse_response(response)
            _cache_put(key, list(entries))
            elapsed = (time.perf_counter() - started) * 1000
            log.info("llm extraction ok notes=%d entries=%d attempt=%d %.0fms", len(notes), len(entries), attempt, elapsed)
            return ExtractionResult(entries, "gemini", elapsed)
        except Exception as exc:  # noqa: BLE001 - provider errors must never escape
            last_error = type(exc).__name__
            log.warning("llm extraction attempt %d failed: %s", attempt, last_error)
            if attempt < settings.llm_max_retries:
                time.sleep(0.25 * (attempt + 1))

    elapsed = (time.perf_counter() - started) * 1000
    log.error("llm extraction exhausted retries (%s) after %.0fms", last_error, elapsed)
    return ExtractionResult([], "failed", elapsed, last_error)
```

### 3.3 Reliability contract for this module

| Failure | Behaviour |
|---|---|
| Missing / invalid API key | `source="unavailable"`, empty entries → guardrails emit `no_op` for every note. HTTP **200**, never 500. |
| Timeout, 429 quota, 5xx from provider | one retry with backoff, then empty entries → `no_op` fallback. |
| Malformed JSON or unknown directive string | parse error caught; entry discarded by guardrails, replaced with `no_op`. |
| Identical notes seen again | served from the in-process LRU cache (bounded by `LLM_CACHE_SIZE`). |

Latency budget: Gemini 2.0 Flash structured extraction for 1–3 short notes typically returns in 0.4–1.2 s; the LP adds ~5–15 ms. That leaves the p95 ≤ 5 s band comfortably. `LLM_TIMEOUT_SECONDS=8` bounds the worst case well inside the 30 s hard timeout.

**Verify Phase 3** (requires a key; skip if absent):

```bash
python - <<'PY'
from app.llm_client import extract_directives
r = extract_directives(["Solar drops to about 20% from 1 PM to 3 PM.", "The cafeteria menu changes tomorrow."], 200)
print(r.source, r.latency_ms)
for e in r.entries: print(e.model_dump())
PY
```

---

## Phase 4 — Deterministic guardrail validation engine

This layer is where interpretation points are defended. It assumes the model may return anything at all — wrong count, duplicate indices, `hours` of `[25, 25, 3]`, a factor of `80`, a reserve of `0.5`, a null adjustment on a real directive — and it emits a response fragment that is *structurally impossible* to fail the schema checks.

Create `app/guardrails.py`:

```python
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


def _clean_explanation(text: str | None, fallback: str) -> str:
    if not text or not str(text).strip():
        return fallback
    cleaned = " ".join(str(text).split())
    return cleaned[:MAX_EXPLANATION_CHARS]


def _normalise_hours(raw: Iterable | None) -> list[int]:
    """Unique ints inside [0, 23], ascending. Anything else is dropped."""
    if not raw:
        return []
    out: set[int] = set()
    for value in raw:
        try:
            if isinstance(value, bool):
                continue
            hour = int(value)
        except (TypeError, ValueError):
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
            if 1.0 < factor <= 100.0:      # model returned a percentage
                factor = factor / 100.0
            factor = min(1.0, max(0.0, factor))
            adjustment = StructuredAdjustment(hours=hours, factor=round(factor, 6))

        elif directive is DirectiveType.MINIMUM_BATTERY_RESERVE:
            reserve = _finite(entry.minimum_energy_kwh)
            if reserve is None:
                validated.append(_no_op(index, "Reserve level could not be resolved; treated as non-applicable."))
                continue
            if entry.reserve_is_fraction_of_capacity or (0.0 < reserve <= 1.0 and capacity > 1.0):
                reserve = reserve * capacity
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
```

**Guardrail audit table — every rule, and where it is enforced**

| Requirement | Enforcement point |
|---|---|
| Exactly N entries, indices 0..N-1 | `_align_entries` + final assertions |
| Duplicate / missing / out-of-range `note_index` | `_align_entries` positional fallback |
| Only the six allowed types | `DirectiveType(...)` conversion, else `no_op` |
| `applies=True` for non-`no_op`, `False` only for `no_op` | hard-coded, never read from the model |
| `structured_adjustment=null` only for `no_op` | `_no_op()` is the single constructor for that shape |
| Hours unique, ints, `[0,23]`, ascending | `_normalise_hours` |
| `factor ∈ [0,1]` | percentage rescue + clamp |
| Reserve finite, ≥0, ≤ capacity | fraction-of-capacity expansion + clamp |
| `max_grid_kwh` finite, ≥0 | `_finite` + sign check |
| Exact key set per directive type | `StructuredAdjustment.model_serializer` |
| No invented demand/solar/tariff/battery | the model has no channel to express them |
| Never crash on bad output | every branch degrades to `no_op` |

**Verify Phase 4:**

```bash
python - <<'PY'
from app.guardrails import apply_deterministic_guardrails as g
from app.schemas import LLMDirectiveEntry
raw = [LLMDirectiveEntry(note_index=5, directive_type="solar_reduction", hours=[14,13,13,99], factor=80)]
out = g(raw, 2, 200.0, 40.0)
print([e.model_dump() for e in out])
assert out[0].structured_adjustment.model_dump() == {"hours":[13,14],"factor":0.8}
assert out[1].directive_type.value == "no_op" and out[1].structured_adjustment is None
print("guardrails ok")
PY
```

---

## Phase 5 — SciPy HiGHS LP solver

### 5.1 Mathematical formulation

Decision vector **x** has 120 entries, 5 per hour *h*:

```
index(h, k) = 5h + k    with k = 0:grid_h  1:solar_used_h  2:charge_h  3:discharge_h  4:E_h
```

**Objective**

```
min  Σ_h  tariff_h · grid_h   +   ε · Σ_h (charge_h + discharge_h),      ε = 1e-6
```

The ε term is a *tie-break only*: with no round-trip losses, simultaneous charge and discharge in the same hour is cost-neutral, and a bare LP may return such a degenerate basis, which cannot be expressed in the single-`battery_action` response schema. The penalty selects the churn-free vertex. Its worst-case distortion of the reported cost is bounded by `ε · Σ(max_charge + max_discharge) ≈ 3×10⁻³ BDT`, two orders of magnitude inside the 0.01 BDT tolerance. **All ten public optima are reproduced exactly with it enabled.**

**Equality system** — 48 rows, `A_eq x = b_eq`:

*Rows 0..23 — hourly energy balance*

```
grid_h + solar_used_h + discharge_h − charge_h = demand_h
```

*Rows 24..47 — battery state transition*

```
h = 0 :  E_0 − charge_0 + discharge_0                    = initial_energy
h > 0 :  E_h − E_{h−1} − charge_h + discharge_h          = 0
```

**Variable bounds** (this is where every directive lands):

| Variable | Lower | Upper |
|---|---|---|
| `grid_h` | 0 | `max_grid_kwh` if hour is in a `max_grid_window`, else `None` (∞) |
| `solar_used_h` | 0 | `effective_solar_h = solar_h · Π factors` |
| `charge_h` | 0 | `0` if hour ∈ `no_charge_window`, else `max_charge_kwh_per_hour` |
| `discharge_h` | 0 | `0` if hour ∈ `no_discharge_window`, else `max_discharge_kwh_per_hour` |
| `E_h` (h<23) | `max(minimum_energy_kwh, reserve_h)` | `capacity_kwh` |
| `E_23` | `initial_energy_kwh` | `initial_energy_kwh` (neutrality as a fixed bound) |

When several `solar_reduction` directives touch one hour the factors multiply; overlapping reserves take the maximum; overlapping grid caps take the minimum. Hour 23's reserve is subsumed by the neutrality equality — if an extracted reserve exceeds `initial_energy_kwh` at hour 23 the problem is infeasible, which the fallback ladder in §5.3 resolves.

### 5.2 `app/directives.py`

```python
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
```

### 5.3 `app/solver.py`

```python
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
```

### 5.4 `app/summary.py`

```python
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
```

**Verify Phase 5** — this must print `ok` for all ten cases *before* any LLM is involved:

```bash
python - <<'PY'
import json
from app.schemas import OptimizeEnergyRequest, DirectiveInterpretationEntry, StructuredAdjustment, DirectiveType
from app.directives import compile_directives
from app.solver import solve_energy_dispatch

pack = json.load(open("data/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"))
for case in pack["cases"]:
    req = OptimizeEnergyRequest.model_validate(case["input"])
    entries = [DirectiveInterpretationEntry.model_validate(e) for e in case["expected_output"]["directive_interpretation"]]
    d = compile_directives(entries, req.solar)
    r = solve_energy_dispatch(req.demand, req.tariff, req.battery, d)
    exp = case["expected_output"]
    assert abs(r.total_cost_bdt - exp["total_cost_bdt"]) <= 0.01, (case["id"], r.total_cost_bdt, exp["total_cost_bdt"])
    assert abs(r.total_grid_kwh - exp["total_grid_kwh"]) <= 0.01
    assert abs(r.peak_grid_kwh - exp["peak_grid_kwh"]) <= 0.01
    print(case["id"], "ok", r.total_cost_bdt)
PY
```

---

## Phase 6 — FastAPI application & routing

Create `app/main.py`:

```python
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    llm_client.warmup()
    log.info("gridwise ready model=%s llm_enabled=%s", settings.gemini_model, settings.llm_enabled)
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
            timeout=settings.llm_timeout_seconds + 2.0,
        )
        raw_entries, source = extraction.entries, extraction.source
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
```

### 6.1 Status-code contract

| Situation | Code | Body |
|---|---|---|
| Healthy service | 200 | `{"status": "ok"}` |
| Valid scenario | 200 | full `OptimizeEnergyResponse` |
| Malformed JSON, missing field, `len(hours) != 24`, empty/oversized `operator_notes` | **400** | `{"error":"invalid_request", ...}` |
| LLM unavailable / timed out / garbage output | **200** with `no_op` interpretations and a still-valid plan | — |
| Anything unexpected | **500** | `{"error":"internal_error","incident_id":"..."}` — no stack trace, no key |

FastAPI's default 422 for validation is deliberately overridden to 400, which the Problem Statement names as the code for structurally invalid requests (422 is optional).

**Verify Phase 6:**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 2
curl -s localhost:8000/health              # {"status":"ok"}
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/optimize-energy \
     -H 'Content-Type: application/json' -d '{"scenario_id":"X"}'   # 400
```

---

## Phase 7 — Automated test suite & public-sample verification

### 7.1 `tests/replay.py` — an independent judge

Written deliberately without importing the solver, so a bug in the solver cannot hide behind a matching bug in the checker.

```python
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
```

### 7.2 `tests/conftest.py`

```python
"""Test fixtures. Default mode stubs the LLM so the suite runs with no API key."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import llm_client
from app.llm_client import ExtractionResult
from app.schemas import LLMDirectiveEntry

SAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


@pytest.fixture(scope="session")
def sample_pack() -> dict:
    return json.loads(SAMPLES_PATH.read_text())


def _expected_to_llm_entries(expected: list[dict]) -> list[LLMDirectiveEntry]:
    """Turn a case's ground-truth interpretation into the flat shape the model would emit."""
    entries = []
    for item in expected:
        adjustment = item.get("structured_adjustment") or {}
        entries.append(
            LLMDirectiveEntry(
                note_index=item["note_index"],
                directive_type=item["directive_type"],
                hours=adjustment.get("hours", []),
                factor=adjustment.get("factor"),
                minimum_energy_kwh=adjustment.get("minimum_energy_kwh"),
                max_grid_kwh=adjustment.get("max_grid_kwh"),
                explanation=item.get("explanation", ""),
            )
        )
    return entries


@pytest.fixture
def client(monkeypatch, sample_pack) -> TestClient:
    """TestClient whose LLM step is replaced by ground truth, unless RUN_LLM_TESTS=1."""
    if os.getenv("RUN_LLM_TESTS") != "1":
        by_notes = {
            tuple(n.strip() for n in case["input"]["operator_notes"]):
            _expected_to_llm_entries(case["expected_output"]["directive_interpretation"])
            for case in sample_pack["cases"]
        }

        def fake_extract(notes, battery_capacity_kwh):
            key = tuple(n.strip() for n in notes)
            return ExtractionResult(list(by_notes.get(key, [])), "stub", 0.0)

        monkeypatch.setattr(llm_client, "extract_directives", fake_extract)

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
```

### 7.3 `tests/test_samples.py` — the headline suite

```python
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
```

### 7.4 `tests/test_api_contract.py`

```python
from __future__ import annotations

import copy


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_body_is_400(client):
    assert client.post("/optimize-energy", json={"scenario_id": "X"}).status_code == 400


def test_twenty_three_hours_is_400(client, sample_pack):
    payload = copy.deepcopy(sample_pack["cases"][0]["input"])
    payload["hours"] = payload["hours"][:23]
    assert client.post("/optimize-energy", json=payload).status_code == 400


def test_duplicate_hour_is_400(client, sample_pack):
    payload = copy.deepcopy(sample_pack["cases"][0]["input"])
    payload["hours"][5]["hour"] = 6
    assert client.post("/optimize-energy", json=payload).status_code == 400


def test_empty_notes_is_400(client, sample_pack):
    payload = copy.deepcopy(sample_pack["cases"][0]["input"])
    payload["operator_notes"] = []
    assert client.post("/optimize-energy", json=payload).status_code == 400


def test_unknown_extra_fields_are_tolerated(client, sample_pack):
    payload = copy.deepcopy(sample_pack["cases"][0]["input"])
    payload["unexpected_field"] = {"anything": True}
    assert client.post("/optimize-energy", json=payload).status_code == 200


def test_error_body_leaks_nothing(client):
    body = client.post("/optimize-energy", json={"bad": 1}).json()
    text = str(body).lower()
    assert "traceback" not in text and "gemini_api_key" not in text and "aiza" not in text
```

### 7.5 `tests/test_guardrails.py`

```python
from __future__ import annotations

from app.guardrails import apply_deterministic_guardrails as guard
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
```

### 7.6 `tests/test_solver.py`

```python
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
```

### 7.7 `tests/test_llm_live.py` — opt-in extraction accuracy

```python
"""Real Gemini calls. Enable with RUN_LLM_TESTS=1 and a valid GEMINI_API_KEY."""
from __future__ import annotations

import os

import pytest

from app.guardrails import apply_deterministic_guardrails
from app.llm_client import extract_directives

pytestmark = pytest.mark.skipif(os.getenv("RUN_LLM_TESTS") != "1", reason="live LLM tests are opt-in")


def test_every_public_case_extracts_ground_truth(sample_pack):
    failures = []
    for case in sample_pack["cases"]:
        notes = case["input"]["operator_notes"]
        capacity = case["input"]["battery"]["capacity_kwh"]
        result = extract_directives(notes, capacity)
        entries = apply_deterministic_guardrails(result.entries, len(notes), capacity,
                                                 case["input"]["battery"]["minimum_energy_kwh"])
        for got, want in zip(entries, case["expected_output"]["directive_interpretation"]):
            if got.directive_type.value != want["directive_type"]:
                failures.append(f"{case['id']} note {got.note_index}: type {got.directive_type.value} != {want['directive_type']}")
                continue
            if want["structured_adjustment"] is None:
                continue
            produced = got.structured_adjustment.model_dump()
            if produced.get("hours") != want["structured_adjustment"]["hours"]:
                failures.append(f"{case['id']} note {got.note_index}: hours {produced.get('hours')} != {want['structured_adjustment']['hours']}")
            for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
                if key in want["structured_adjustment"]:
                    if abs(produced.get(key, -1) - want["structured_adjustment"][key]) > 0.01:
                        failures.append(f"{case['id']} note {got.note_index}: {key} {produced.get(key)} != {want['structured_adjustment'][key]}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("note,expected_hours,expected_factor", [
    ("PV production will drop to about 20% between 13:00 and 15:00.", [13, 14], 0.2),
    ("Panel washing from one until three will leave roughly one-fifth of normal solar output.", [13, 14], 0.2),
    ("Expect an 80% reduction in rooftop solar during the 1-3 PM maintenance window.", [13, 14], 0.2),
])
def test_paraphrase_robustness(note, expected_hours, expected_factor):
    entries = apply_deterministic_guardrails(extract_directives([note], 200.0).entries, 1, 200.0, 40.0)
    adjustment = entries[0].structured_adjustment.model_dump()
    assert entries[0].directive_type.value == "solar_reduction"
    assert adjustment["hours"] == expected_hours
    assert abs(adjustment["factor"] - expected_factor) <= 0.01
```

### 7.8 Running the suite

```bash
pytest -q                      # offline: guardrails + solver + contract + all 10 samples
RUN_LLM_TESTS=1 pytest -q      # adds real Gemini extraction accuracy + paraphrase robustness
```

Everything except `test_llm_live.py` must pass with **no** API key set. Do not proceed to Phase 8 until `pytest -q` is green.

---

## Phase 8 — Dockerization & public ingress

### 8.1 `Dockerfile`

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

No secret is ever `COPY`-ed or `ENV`-ed in; `.dockerignore` excludes `.env`. Verify with `docker history gridwise:latest` before pushing.

### 8.2 Build, run, publish

```bash
# Build
docker build -t gridwise:latest .

# Run (key injected at runtime only)
docker run --rm -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" --name gridwise gridwise:latest

# Or from a file that is never committed
docker run --rm -p 8000:8000 --env-file .env --name gridwise gridwise:latest

# Verify
curl -s localhost:8000/health
curl -s -X POST localhost:8000/optimize-energy -H 'Content-Type: application/json' \
     -d @data/sample_request.json | head -c 400

# Publish a pullable fallback image with an exact tag, then record the digest
docker tag gridwise:latest docker.io/<dockerhub-user>/gridwise:preli-1.0.0
docker push docker.io/<dockerhub-user>/gridwise:preli-1.0.0
docker inspect --format='{{index .RepoDigests 0}}' docker.io/<dockerhub-user>/gridwise:preli-1.0.0
```

Extract `data/sample_request.json` once from the public pack:

```bash
python -c "import json;p=json.load(open('data/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json'));json.dump(p['cases'][0]['input'],open('data/sample_request.json','w'),indent=2)"
```

Confirm the image is secret-free and starts clean from a cold pull:

```bash
docker rmi docker.io/<dockerhub-user>/gridwise:preli-1.0.0
docker pull docker.io/<dockerhub-user>/gridwise:preli-1.0.0
docker run --rm -d -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" docker.io/<dockerhub-user>/gridwise:preli-1.0.0
sleep 5 && curl -s localhost:8000/health
```

### 8.3 Cloudflare Tunnel — zero-cost public HTTPS

```bash
# Install (Debian/Ubuntu)
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb

# With the service already listening on :8000
cloudflared tunnel --no-autoupdate --url http://localhost:8000
```

`cloudflared` prints a `https://<random-words>.trycloudflare.com` URL. That is the submitted base URL. Operating notes:

- Keep the tunnel process alive for the whole judging window (`tmux`, `screen`, or `systemd`). Killing it invalidates the URL.
- A quick tunnel has no cold start and no login, which is exactly what the "no login, no manual approval" rule requires.
- Immediately smoke-test from **outside** the dev machine:

```bash
curl -s https://<your-tunnel>.trycloudflare.com/health
curl -s -X POST https://<your-tunnel>.trycloudflare.com/optimize-energy \
     -H 'Content-Type: application/json' -d @data/sample_request.json | python -m json.tool | head -40
```

- Run a quick p95 check against the public URL before submitting:

```bash
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{time_total}\n' -X POST https://<your-tunnel>.trycloudflare.com/optimize-energy \
       -H 'Content-Type: application/json' -d @data/sample_request.json
done | sort -n | awk '{a[NR]=$1} END {print "p50="a[int(NR*0.5)], "p95="a[int(NR*0.95)]}'
```

If p95 exceeds 5 s, lower `LLM_TIMEOUT_SECONDS`, confirm the cache is enabled, and re-check the model name.

---

## Phase 9 — `README.md` specification

The README is scored on a fixed checklist worth 10 points: 3 for a clean quickstart from a fresh environment, 2 for environment/model-provider documentation, 2 for the public-sample test procedure and expected result, 1 for the LLM → guardrail → optimizer architecture, 1 for Docker pull/run fallback, 1 for dependencies/limitations/secret handling. Write it to hit each line explicitly.

Required section order and content:

1. **Title + one-paragraph overview** — what the service does, and that a language model performs the operator-note interpretation that feeds the optimizer.
2. **Architecture** — the flow `Request → Gemini 2.0 Flash structured extraction → deterministic guardrails → directive compilation → SciPy HiGHS LP → replay self-check → response`, with one line per module and a statement that the LLM sits on the interpretation path, not just on `plan_summary`.
3. **Quickstart (fresh clone)** — copy-pasteable, no undocumented steps:
   ```bash
   git clone <repo-url> && cd gridwise
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env        # then set GEMINI_API_KEY
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
4. **Configuration** — a table of every environment variable *name* (`GEMINI_API_KEY`, `GEMINI_MODEL`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_ENABLED`, `LLM_CACHE_SIZE`, `LOG_LEVEL`, `PORT`) with purpose, default, and required/optional. **Values are never printed.** State where a free Google AI Studio key comes from.
5. **Model & solver disclosure** — provider *Google AI Studio*, model `gemini-2.0-flash` via the official `google-genai` SDK, structured outputs with `response_schema` + `response_mime_type="application/json"` at `temperature=0.0`; solver `scipy.optimize.linprog(method="highs")`, a 120-variable continuous LP.
6. **Endpoints** — `GET /health` and `POST /optimize-energy` with full request and response JSON examples and working curl commands:
   ```bash
   curl -s http://localhost:8000/health
   curl -s -X POST http://localhost:8000/optimize-energy \
        -H 'Content-Type: application/json' \
        -d @data/sample_request.json
   ```
   Include one real, complete response body (trimmed `hourly_plan` is acceptable if clearly marked).
7. **Public-sample verification** — the exact command and the exact expected outcome:
   ```bash
   pytest -q                 # 10/10 public cases pass with no API key required
   RUN_LLM_TESTS=1 pytest -q # additionally verifies live Gemini extraction accuracy
   ```
   Reproduce the benchmark table from §0.2 so judges can see the expected costs.
8. **Guardrails** — the audit table from Phase 4: what is checked, what is clamped, what degrades to `no_op`, and the statement that `applies`/`structured_adjustment` shape are derived deterministically, never trusted from the model.
9. **Docker fallback** — exact `docker pull` / `docker run` commands with the published tag **and** digest, the exposed port, the required env var name, and a note that the image has no baked-in secrets.
10. **Public endpoint** — the live base URL plus the two curl commands against it.
11. **Dependencies & credits** — FastAPI, Uvicorn, Pydantic, `google-genai`, SciPy (HiGHS), NumPy, pytest, cloudflared; each with its role and licence family.
12. **Known limitations** — quick-tunnel URL is ephemeral; free-tier quota and rate limits; `no_op` degradation when the provider is unreachable (service stays up and still returns a valid plan); in-process cache is not shared across replicas; ambiguous notes resolve to `no_op` by design.
13. **Security & secret handling** — `.env` is git-ignored, no secrets in the image or logs, logs are redacted, error responses carry an incident id instead of a stack trace, only synthetic challenge data is used.

**Final README sanity check:** on a clean machine with no prior context, follow only the README. If any step requires knowledge that is not written down, the README is incomplete.

---

## Phase 10 — Final acceptance gate

Run every line. Do not submit until all pass.

```bash
# 1. Offline suite: guardrails, solver, contract, all 10 public cases
pytest -q

# 2. Live extraction accuracy (needs key)
RUN_LLM_TESTS=1 pytest -q tests/test_llm_live.py

# 3. Secret scan
git ls-files | xargs grep -nEi "AIza[0-9A-Za-z_-]{10,}|api[_-]?key\s*[:=]\s*['\"][^'\"]+" || echo "no secrets committed"
git check-ignore -v .env

# 4. Container: builds, starts as non-root, healthy, no baked secrets
docker build -t gridwise:latest .
docker run --rm -d -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" --name gw gridwise:latest
sleep 5 && curl -s localhost:8000/health && docker exec gw whoami
docker run --rm --entrypoint sh gridwise:latest -c 'env | grep -i gemini || echo "no secret in image env"'
docker stop gw

# 5. Public endpoint reachable from outside
curl -s https://<your-tunnel>.trycloudflare.com/health
curl -s -X POST https://<your-tunnel>.trycloudflare.com/optimize-energy \
     -H 'Content-Type: application/json' -d @data/sample_request.json | head -c 300

# 6. Latency
for i in $(seq 1 20); do curl -s -o /dev/null -w '%{time_total}\n' -X POST \
  https://<your-tunnel>.trycloudflare.com/optimize-energy -H 'Content-Type: application/json' \
  -d @data/sample_request.json; done | sort -n | awk '{a[NR]=$1} END {print "p95="a[int(NR*0.95)]}'
```

**Submission checklist**

- [ ] `GET /health` returns `{"status":"ok"}` on the public URL.
- [ ] `POST /optimize-energy` returns the exact response contract on the public URL.
- [ ] Exactly one `directive_interpretation` entry per note, in `note_index` order; `no_op` ⇒ `applies:false` + `null`; all others ⇒ `applies:true` + exact adjustment shape.
- [ ] All 10 public cases pass the replay validator and match the reference optimal cost within 0.01 BDT.
- [ ] Totals recomputed from `hourly_plan`; battery neutral at hour 23.
- [ ] Repo created after question reveal, private during the event, public after the deadline.
- [ ] README self-contained; no secret values anywhere.
- [ ] Docker image pushed with exact tag **and** digest recorded; pulls and reaches `/health` with the documented command.
- [ ] 3-minute video covering problem, architecture, LLM → guardrails → optimizer flow, and how to run/test (tie-break only, no base points).
- [ ] Tunnel process supervised and stays up for the whole judging window.

---

## Appendix A — Time expression normalization reference

| Wording | Hours |
|---|---|
| "1 PM to 3 PM", "13:00–15:00", "one until three (afternoon)" | `[13, 14]` |
| "between 11 AM and 2 PM" | `[11, 12, 13]` |
| "from noon until 2 PM" | `[12, 13]` |
| "2 AM until 5 AM" | `[2, 3, 4]` |
| "6 PM until 9 PM" | `[18, 19, 20]` |
| "6 PM until 10 PM" | `[18, 19, 20, 21]` |
| "during the 5 PM hour" | `[17]` |
| "from 6 PM onwards" / "rest of the evening" | `[18 … 23]` |
| "all day" / "throughout the day" | `[0 … 23]` |
| "10 PM until 2 AM" (wraps midnight) | `[0, 1, 22, 23]` (ascending) |

## Appendix B — Solar factor normalization reference

| Wording | `factor` |
|---|---|
| "drops to about 20%" | 0.20 |
| "80% reduction" | 0.20 |
| "roughly one-fifth of normal" | 0.20 |
| "reduced to 25% of forecast" | 0.25 |
| "about half the forecast" | 0.50 |
| "cut to one third" | 0.33 |
| "panels fully covered", "no usable solar" | 0.00 |
| "30% lower than forecast" | 0.70 |

## Appendix C — Failure-mode matrix

| Failure | Detection | Response |
|---|---|---|
| Missing API key | `_get_client()` returns `None` | 200, all notes `no_op`, valid plan, error logged |
| Provider timeout / 429 / 5xx | exception in `generate_content` | one retry, then `no_op` fallback; still 200 |
| Model returns 2 entries for 3 notes | `_align_entries` | missing slot filled with `no_op` |
| Model returns `hours: [25, 25, 3]` | `_normalise_hours` | `[3]` |
| Model returns `factor: 80` | percentage rescue | `0.8` |
| Model returns `factor: 1.4` | clamp | `1.0` |
| Model returns reserve `0.5` with capacity 200 | fraction expansion | `100.0` |
| Model returns reserve above capacity | clamp | `capacity` |
| Model invents `"grid_export"` | enum conversion fails | `no_op` |
| Directives make the LP infeasible | `linprog` failure | relaxation ladder, then grid-only plan; always 24 valid hours |
| Degenerate simultaneous charge + discharge | ε churn penalty + netting in `_build_plan` | single legal `battery_action` per hour |
| Float drift at hour 23 | snap in `_build_plan` | exact neutrality |
| Malformed request payload | `RequestValidationError` handler | 400 with field list, no trace |
| Any unexpected exception | global handler | 500 with incident id, no trace, no key |
