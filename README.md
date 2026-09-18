# GridWise Energy Optimization Service

A FastAPI microservice for the BUP CSE Fest 2026 preliminary (GridWise challenge). It takes a
24-hour campus energy scenario plus short free-text **operator notes**, and returns a cost-optimal
hourly dispatch plan for grid, solar and battery.

**A language model does the interpretation.** Each operator note is read by a Gemini model that
extracts a structured directive (solar reduction, minimum battery reserve, no-charge window,
no-discharge window, hourly grid cap, or `no_op`). That extraction is what feeds the optimizer, not
just the `plan_summary` text. Because model output is untrusted, it passes through a deterministic
guardrail layer before it ever reaches the maths, and the dispatch itself is solved exactly as a
linear program.

---

## 1. Architecture

```
POST /optimize-energy
   │
   ▼
Request validation (Pydantic; malformed -> 400)
   │
   ▼
Gemini structured extraction  ── model chain, hedging, cooldowns, cache (app/llm_client.py)
   │   one flat entry per operator note, JSON-schema constrained
   ▼
Deterministic guardrails      ── validate / clamp / normalise; anything doubtful -> no_op
   │
   ▼
Directive compilation         ── effective solar, reserve floors, grid caps, blackout windows
   │
   ▼
SciPy HiGHS linear program    ── 120 variables, exact minimum-cost dispatch
   │
   ▼
Replay self-check + summary   ── energy balance / battery transition re-verified, totals recomputed
   │
   ▼
Response (200, exact contract)
```

The LLM sits on the **interpretation path**: without its output no directive reaches the solver.

| Module | Role |
|---|---|
| `app/main.py` | FastAPI app, routes, 400/500 error handlers, startup warm-up |
| `app/schemas.py` | Every Pydantic contract: request, flat LLM schema, response |
| `app/prompts.py` | System instruction, worked examples, note rendering |
| `app/llm_client.py` | Gemini calls: fallback chain, hedged requests, per-model cooldowns, cache, never raises |
| `app/guardrails.py` | Deterministic audit of the model's output (see section 8) |
| `app/directives.py` | Compiles validated directives into optimizer inputs |
| `app/solver.py` | HiGHS LP, relaxation ladder for infeasible directive sets, plan post-processing |
| `app/summary.py` | Deterministic `plan_summary` text |
| `app/config.py` | Environment-driven settings; no secret has a default |
| `app/logging_utils.py` | Logger that redacts credentials |

---

## 2. Quickstart (fresh clone)

Requires Python 3.11 to 3.13 and a free Google AI Studio API key (see section 3).

Linux / macOS:

```bash
git clone https://github.com/arham-apon/Campus-light.git && cd Campus-light
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env and set GEMINI_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Windows (PowerShell):

```powershell
git clone https://github.com/arham-apon/Campus-light.git; cd Campus-light
py -3.13 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then edit .env and set GEMINI_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check it:

```bash
curl -s http://localhost:8000/health
```

`requirements.txt` is the install surface. `requirements.lock.txt` is a `pip freeze` of the exact
environment the code was developed and tested in (Windows, Python 3.13) if you want a byte-exact
reproduction.

A `Makefile` wraps the same steps: `make install`, `make run`, `make test`, `make docker-build`.

---

## 3. Configuration

Set in the environment or in a `.env` file (copy `.env.example`; `.env` is git-ignored). Values are
never printed by the service.

| Variable | Purpose | Default | Required |
|---|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio key. Free key: <https://aistudio.google.com/apikey> | none | **Yes** for LLM interpretation. Without it the service stays up but every note degrades to `no_op` |
| `GEMINI_MODEL` | Primary model | `gemini-3.5-flash-lite` | No |
| `GEMINI_FALLBACK_MODELS` | Comma-separated models tried in order when the primary is rate-limited, overloaded, slow or gone. Empty disables fallbacks | `gemini-3.5-flash,gemini-3.1-flash-lite,gemma-4-26b-a4b-it` | No |
| `LLM_TIMEOUT_SECONDS` | Per-call timeout. The Gemini API rejects anything under 10 s, so lower values are raised to 10 | `10` | No |
| `LLM_HEDGE_DELAY_SECONDS` | If a model has not answered after this long the next healthy one starts in parallel (`0` = never) | `2.5` | No |
| `LLM_TOTAL_BUDGET_SECONDS` | Hard cap on LLM time per request (clamped to 5-25); beyond it notes degrade to `no_op` | `20` | No |
| `LLM_MAX_RETRIES` | Extra passes over the model chain after every model failed | `1` | No |
| `LLM_ENABLED` | Set `false` to skip the LLM entirely (all notes `no_op`) | `true` | No |
| `LLM_CACHE_SIZE` | In-process LRU entries for repeated notes (`0` disables) | `512` | No |
| `LOG_LEVEL` | Python log level | `INFO` | No |
| `PORT` | Documented listen port. The commands and the Docker image listen on `8000`; pass `--port` to `uvicorn` to change it | `8000` | No |

---

## 4. Model and solver disclosure

- **Provider:** Google AI Studio (Gemini API), through the official `google-genai` SDK.
- **Primary model:** `gemini-3.5-flash-lite`. Structured output is requested with
  `response_schema` (a flat Pydantic model) and `response_mime_type="application/json"` at
  `temperature=0.0`.
- **Fallback chain:** `gemini-3.5-flash` (minimal thinking), `gemini-3.1-flash-lite`,
  `gemma-4-26b-a4b-it`. They exist because free-tier quotas are enforced per model (see section 12),
  so a chain adds both resilience and capacity. On the ten public cases the primary extracted all 18
  notes correctly; the last-resort Gemma model got 15 of 18.
- **Solver:** `scipy.optimize.linprog(method="highs")`, a 120-variable continuous LP (5 variables per
  hour: grid, solar used, charge, discharge, stored energy) with 48 equality constraints.

---

## 5. Endpoints

### `GET /health`

Never touches the LLM or the solver, so readiness is immediate.

```bash
curl -s http://localhost:8000/health
# {"status":"ok"}
```

### `POST /optimize-energy`

```bash
curl -s -X POST http://localhost:8000/optimize-energy \
     -H 'Content-Type: application/json' \
     -d @data/sample_request.json
```

Request (abridged; `data/sample_request.json` has the full 24 hours):

```jsonc
{
  "scenario_id": "SAMPLE-01",
  "operator_notes": [
    "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
    "The sports office moved next month's registration deadline."
  ],
  "hours": [
    { "hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6 }
    // ... exactly 24 entries, hour 0..23 once each
  ],
  "battery": {
    "capacity_kwh": 220, "initial_energy_kwh": 110, "minimum_energy_kwh": 40,
    "max_charge_kwh_per_hour": 50, "max_discharge_kwh_per_hour": 50
  }
}
```

Response for `data/sample_request.json` (`hourly_plan` trimmed here to 6 of its 24 entries; the real
response has all 24. The `explanation` wording is written by the model and varies between runs):

```jsonc
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": { "hours": [12, 13], "factor": 0.25 },
      "explanation": "Solar availability is reduced to 25% during the panel-cleaning window."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "This note does not affect today's 24-hour energy schedule."
    }
  ],
  "hourly_plan": [
    { "hour": 0,  "grid_kwh": 90.0,  "solar_used_kwh": 0.0,   "battery_action": "idle",      "battery_kwh": 0.0,  "battery_energy_after_kwh": 110.0 },
    { "hour": 1,  "grid_kwh": 45.0,  "solar_used_kwh": 0.0,   "battery_action": "discharge", "battery_kwh": 40.0, "battery_energy_after_kwh": 70.0 },
    { "hour": 2,  "grid_kwh": 130.0, "solar_used_kwh": 0.0,   "battery_action": "charge",    "battery_kwh": 50.0, "battery_energy_after_kwh": 120.0 },
    // ... hours 3-12 omitted ...
    { "hour": 13, "grid_kwh": 152.5, "solar_used_kwh": 42.5,  "battery_action": "charge",    "battery_kwh": 15.0, "battery_energy_after_kwh": 120.0 },
    { "hour": 14, "grid_kwh": 80.0,  "solar_used_kwh": 140.0, "battery_action": "charge",    "battery_kwh": 50.0, "battery_energy_after_kwh": 170.0 },
    // ... hours 15-22 omitted ...
    { "hour": 23, "grid_kwh": 155.0, "solar_used_kwh": 0.0,   "battery_action": "charge",    "battery_kwh": 50.0, "battery_energy_after_kwh": 110.0 }
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "Applied reduced solar availability. 1 note(s) were non-operational and treated as no_op. The plan buys grid energy in the cheapest feasible hours, charges the battery in 8 hour(s) and discharges it in 8 hour(s) to cover expensive periods, uses available solar before grid import, and returns the battery to its starting state of charge by hour 23. Total grid cost is 38365.00 BDT with a peak hourly import of 175.00 kWh."
}
```

Totals are recomputed from `hourly_plan`, which is the source of truth, and the battery ends the day
at its starting energy (110 kWh here).

| Situation | Status | Body |
|---|---|---|
| Valid scenario | 200 | Full response above |
| Malformed JSON, missing field, `len(hours) != 24`, duplicate hour, empty or more than 3 notes | **400** | `{"error":"invalid_request","fields":[...]}` |
| LLM unavailable, timed out, or returned garbage | **200** | All affected notes `no_op`, still a valid plan |
| Anything unexpected | **500** | `{"error":"internal_error","incident_id":"..."}`: no stack trace, no key |

---

## 6. Public-sample verification

```bash
pytest -q                    # offline: no API key and no network needed
RUN_LLM_TESTS=1 pytest -q    # additionally runs live Gemini extraction accuracy + paraphrase tests
```

The default run replaces only the LLM step with each case's ground-truth interpretation, and runs
everything else for real. All ten public cases go through the full HTTP stack, an independent replay
validator (`tests/replay.py`, which does not import the solver) re-checks every rule hour by hour,
and the cost must match the reference optimum within 0.01 BDT. As delivered: **111 passed, 4 skipped**
(the four skipped tests are the opt-in live-LLM ones).

Expected results the ten public cases must reproduce:

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

## 7. How the LLM layer stays reliable

`app/llm_client.py` treats the provider as unreliable and never lets it break a request:

- **Fallback chain.** Models are tried in order; a failure starts the next model immediately.
- **Hedging.** If the current model has not answered after `LLM_HEDGE_DELAY_SECONDS`, the next
  healthy model starts alongside it and the first usable answer wins.
- **Cooldowns.** A rate-limited or failing model is skipped for a while (a per-minute quota for the
  provider's retry hint, a per-day quota for an hour, a retired model for five minutes) so later
  requests do not waste a call on it.
- **Budget.** A request never waits more than `LLM_TOTAL_BUDGET_SECONDS` on the LLM.
- **Cache.** Identical notes are served from an in-process LRU cache.
- **Warm-up.** The SDK import, client and first TLS connection are prepared in the background at
  startup, so the first request is not slower than the rest. `/health` never waits for it.
- **Degradation.** If nothing answers, the response is still `200` with `no_op` for the affected
  notes and a valid plan.

---

## 8. Guardrails

The model's output is never trusted. `applies` and the exact shape of `structured_adjustment` are
**derived deterministically** by the guardrail layer; the model is not even asked for `applies`.

| Requirement | Enforced by |
|---|---|
| Exactly one entry per note, `note_index` 0..N-1 in order | Alignment by index, positional fallback, final assertions |
| Duplicate, missing or out-of-range `note_index` | Positional realignment; a missing slot becomes `no_op` |
| Only the six supported directive types | Enum conversion, otherwise `no_op` |
| `applies` true for every non-`no_op` type, false only for `no_op` | Hard-coded, never read from the model |
| `structured_adjustment` is `null` only for `no_op` | A single constructor for that shape |
| `hours` unique integers in 0..23, ascending | Normalisation drops anything else; no usable hours means `no_op` |
| `factor` in 0..1 (fraction that **remains** usable) | Values 2 to 100 are read as percentages and divided by 100; anything else is clamped to 0..1 |
| Reserve finite, at least 0, at most battery capacity | A share of capacity is expanded (a flagged percentage such as `50` is rescaled first), then clamped |
| `max_grid_kwh` finite and not negative | Otherwise `no_op` |
| Exact key set per directive type | Custom serializer on `StructuredAdjustment` |
| No invented demand, solar, tariff or battery values | The model has no channel to express them |
| Malformed model output can never crash a request | Every branch degrades to `no_op` |

Time windows are whole hours, **start-inclusive and end-exclusive** ("1 PM to 3 PM" is `[13, 14]`).

---

## 9. Docker fallback

The image has no baked-in secrets: `.dockerignore` excludes `.env`, and the key is supplied only at
run time. It runs as a non-root user and exposes port **8000**.

Build and run locally:

```bash
docker build -t gridwise:latest .
docker run --rm -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" --name gridwise gridwise:latest
# or, from a file that is never committed:
docker run --rm -p 8000:8000 --env-file .env --name gridwise gridwise:latest
curl -s localhost:8000/health
```

Pull the published image instead of building:

```bash
docker pull ghcr.io/arham-apon/gridwise:preli-1.0.0
docker run --rm -d -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" ghcr.io/arham-apon/gridwise:preli-1.0.0
sleep 5 && curl -s localhost:8000/health
```

Pinned by immutable digest (identical image, tag-proof):

```bash
docker pull ghcr.io/arham-apon/gridwise@sha256:3bb1ba246734d0b93ba751493dc679b9424be996342293c9cd2535734d756303
docker run --rm -d -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  ghcr.io/arham-apon/gridwise@sha256:3bb1ba246734d0b93ba751493dc679b9424be996342293c9cd2535734d756303
```

The package is public, so both commands work without `docker login`. Verified on the published
image: platform `linux/amd64`, runs as the non-root user `appuser`, exposes port 8000, carries a
`/health` HEALTHCHECK, and has **no Gemini variable of any kind in its environment**.

The image is built, smoke-tested and published by
[`.github/workflows/docker-image.yml`](.github/workflows/docker-image.yml) on every push to `main`.
That workflow starts the published image, asserts `/health` returns `{"status":"ok"}`, asserts it runs
as the non-root user `appuser`, posts a real scenario and checks a 24-hour plan comes back, and
confirms no Gemini secret is baked into the image.

The only required environment variable is `GEMINI_API_KEY`, supplied at run time. The service is
single-process by design (`--workers 1`) because the cache and model cooldowns live in process memory.

---

## 10. Public endpoint

Base URL: **`https://campus-light.onrender.com`**

```bash
curl -s https://campus-light.onrender.com/health
```

```bash
curl -s -X POST https://campus-light.onrender.com/optimize-energy \
     -H 'Content-Type: application/json' -d @data/sample_request.json
```

The service runs on Render from the published GHCR image — it does not depend on any development
machine being switched on. Verified against this URL with
[`scripts/verify_endpoint.py`](scripts/verify_endpoint.py), which replays every returned plan
hour by hour through the independent validator in `tests/replay.py`:

| Check | Result |
|---|---|
| Public sample cases passed | **10 / 10** |
| Operator notes interpreted correctly | **18 / 18** |
| Latency | p50 1.69 s, **p95 2.97 s**, max 2.97 s (limit 30 s, target p95 ≤ 5 s) |
| `GET /health` | 200 `{"status":"ok"}` in 0.46 s |
| Malformed body | 400 |

Reproduce it yourself against the live service:

```bash
python scripts/verify_endpoint.py https://campus-light.onrender.com
```

---

## 11. Dependencies and credits

| Component | Role | Licence family |
|---|---|---|
| FastAPI | HTTP framework, routing, validation hooks | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| Pydantic | Request, response and LLM-schema contracts | MIT |
| google-genai | Official Gemini API SDK | Apache-2.0 |
| SciPy (HiGHS) | `linprog` LP solver | BSD-3-Clause (HiGHS: MIT) |
| NumPy | Constraint matrices | BSD-3-Clause |
| httpx | HTTP client (SDK and tests) | BSD-3-Clause |
| python-dotenv | `.env` loading | BSD-3-Clause |
| pytest | Test runner | MIT |
| cloudflared | Quick tunnel for the public URL | Apache-2.0 |

---

## 12. Known limitations

- **Free-tier quotas.** On a free Google AI Studio key each model is limited to **15 requests per
  minute**, and `gemini-3.5-flash` to **20 requests per day**. Sustained bursts spill onto the
  fallback models, which are slower (roughly 4 to 7 s a request, against about 1.5 s for the primary)
  and, for the last-resort Gemma model, less accurate. A billing-enabled key removes the ceiling.
- **Provider outage.** If every model is unreachable, notes degrade to `no_op` (the service stays up
  and still returns a valid plan, but the directives are not applied).
- **Ambiguous notes** resolve to `no_op` by design rather than guessing.
- **Model availability changes.** Google retires models; set `GEMINI_MODEL` /
  `GEMINI_FALLBACK_MODELS` if a default stops working.
- **In-process state.** The cache and model cooldowns are not shared across replicas or workers.
- **Quick-tunnel URLs are ephemeral.** They change on every `cloudflared` start.
- **Infeasible directive sets** (for example a grid cap too low to meet demand) are handled by a fixed
  relaxation ladder, and as a last resort a grid-only plan, so a valid plan is always returned.

---

## 13. Security and secret handling

- `.env` is git-ignored (`.gitignore`) and excluded from the image (`.dockerignore`). Only
  `.env.example`, which has a placeholder, is committed.
- No secret is in the image, its environment, or the repository. The key is read from the
  environment at run time.
- Logs are redacted: the configured key and Google API-key-shaped tokens are masked before printing.
- Error responses carry an incident id instead of a stack trace and never include credentials.
- The container runs as a non-root user.
- Only the synthetic challenge data is used; no personal data is processed or stored.
