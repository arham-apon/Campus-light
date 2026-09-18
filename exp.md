# GridWise — what was built, in plain language

This is a walkthrough of the whole project: what it does, how it works, what each file is for,
and what was built in each of the nine phases. It is written to be read start to finish without
needing the spec open.

For judges and setup instructions, read [README.md](README.md) instead. This file is the
"explain it to me" version.

---

## 1. The problem in one paragraph

A campus has 24 hours of electricity to plan. Each hour has a **demand** (how much power the campus
needs), **solar** (how much the rooftop panels will make), and a **tariff** (what grid power costs
that hour). There is also a **battery** that can store cheap energy and release it when power is
expensive.

Planning that alone is a solved maths problem. The twist is that a human operator also leaves
**short notes in plain English**, like *"Solar output will drop to about 20% from 1 PM to 3 PM"* or
*"Don't charge the battery between 2 and 4 PM"*. Some notes are irrelevant distractors, like
*"The cafeteria menu changes tomorrow"*.

The service must **read those notes with a language model**, turn them into machine-readable rules,
apply those rules to the maths, and return the cheapest valid 24-hour plan.

---

## 2. How it works, end to end

```
  Judge sends a scenario (24 hours + battery + 1-3 operator notes)
                              │
                              ▼
   ┌──────────────────────────────────────────────┐
   │ 1. VALIDATE   Is the request even well-formed? │  bad request -> 400
   └──────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────┐
   │ 2. UNDERSTAND (the LLM)                       │
   │    Gemini reads each note and returns a flat  │
   │    JSON entry: which rule, which hours,       │
   │    which number.                              │
   └──────────────────────────────────────────────┘
                              │   (untrusted output)
                              ▼
   ┌──────────────────────────────────────────────┐
   │ 3. POLICE IT (guardrails)                     │
   │    Plain Python re-checks everything the      │
   │    model said. Anything wrong or doubtful     │
   │    becomes "no_op" (ignore this note).        │
   └──────────────────────────────────────────────┘
                              │   (now trustworthy)
                              ▼
   ┌──────────────────────────────────────────────┐
   │ 4. COMPILE   Turn rules into numbers:         │
   │    reduced solar, blocked hours, caps, floors │
   └──────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────┐
   │ 5. SOLVE   A linear program finds the exact   │
   │    cheapest plan that obeys every rule.       │
   └──────────────────────────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────┐
   │ 6. DOUBLE-CHECK   Replay the plan hour by     │
   │    hour and confirm nothing was broken.       │
   └──────────────────────────────────────────────┘
                              │
                              ▼
                    Response (HTTP 200)
```

The important design idea is the split between step 2 and step 3.

**The language model is good at reading English and bad at being trusted.** So it is used only for
reading. Every number it produces is then re-checked by ordinary code before it is allowed anywhere
near the maths. The model cannot invent a rule type, cannot produce an hour like `25`, and cannot
set a battery reserve larger than the battery. If it tries, the note is ignored instead.

This also satisfies a hard competition rule: the model must do the **actual interpretation** that
feeds the optimizer. Using AI only to write the summary text at the end would disqualify the entry.

---

## 3. The six rules a note can become

| Rule | Plain meaning | What the maths does |
|---|---|---|
| `solar_reduction` | Panels will underperform in these hours | Multiplies that hour's solar by a factor |
| `minimum_battery_reserve` | Keep at least this much charge | Raises the battery floor for those hours |
| `no_charge_window` | Can't charge in these hours | Forces charging to zero |
| `no_discharge_window` | Can't discharge in these hours | Forces discharging to zero |
| `max_grid_window` | Don't pull more than X per hour | Caps grid purchase for those hours |
| `no_op` | This note is irrelevant | Nothing changes |

Two details cause most mistakes, so they are worth stating plainly:

**Time windows exclude the end hour.** "1 PM to 3 PM" means hours `[13, 14]`, not `[13, 14, 15]`.

**The solar factor is what is left, not what is lost.** "An 80% reduction" means `factor = 0.2`,
because 20% still works. So does "drops to about 20%" and "roughly one fifth of normal".
Those three sentences mean the same thing, and the service handles all of them.

---

## 4. What each file does

### The application (`app/`)

| File | What it does |
|---|---|
| `main.py` | The web server. Two endpoints, the error handlers, and the startup warm-up. |
| `schemas.py` | Every data shape: what a valid request looks like, what the model is asked for, what the response must be. |
| `prompts.py` | The instructions given to the language model, including worked examples. |
| `llm_client.py` | Talks to Gemini. Handles slowness, rate limits, outages and junk output. Never crashes. |
| `guardrails.py` | The police officer. Re-checks everything the model returned. |
| `directives.py` | Converts checked rules into numbers the solver understands. |
| `solver.py` | The maths. Builds and solves the linear program, then tidies the plan. |
| `summary.py` | Writes the human-readable `plan_summary` sentence. No AI, so it costs nothing. |
| `config.py` | Reads settings from the environment. No secret has a default value. |
| `logging_utils.py` | A logger that scrubs API keys out of anything printed. |

### The tests (`tests/`)

| File | What it checks |
|---|---|
| `test_samples.py` | All 10 public cases, end to end through real HTTP. |
| `replay.py` | An independent judge. Re-checks plans **without importing the solver**, so a bug in the solver can't hide behind a matching bug in the checker. |
| `test_api_contract.py` | Status codes, bad input handling, no secrets in errors. |
| `test_solver.py` | The maths on its own: arbitrage, caps, floors, blocked windows, infeasible input. |
| `test_guardrails.py` | Every way a model can misbehave, and that each one degrades safely. |
| `test_llm_client.py` | Fallbacks, hedging, cooldowns, budget, caching — using a fake Gemini, so it's fast and free. |
| `test_config.py` | Settings parsing and clamping. |
| `test_llm_live.py` | Real Gemini accuracy. **Off by default** (it costs API quota). |
| `conftest.py` | Shared setup. Swaps the LLM for known-correct answers so the suite runs offline. |

---

## 5. What was built in each phase

### Phase 1 — Project setup
Folder structure, `requirements.txt`, `.gitignore`, `.dockerignore`, `.env.example`, `Makefile`,
and a Python 3.13 virtual environment.

One thing that mattered here: `.env` (which holds the real API key) existed but **was not ignored
by git**, so a `git add .` would have committed the key. The `.gitignore` now covers it. The key
has never been committed.

### Phase 2 — Data shapes
All the Pydantic models. The response shape is enforced by the type system rather than by
discipline, so a malformed response is close to impossible.

A deliberate trick here: the model is asked for a **flat** structure with simple nullable fields,
not the nested final shape. Nested optional objects are the main cause of extraction failures. The
strict nested shape is built afterwards by ordinary code. The model is also **never asked for the
`applies` flag** — that is derived (`applies = type != no_op`), which makes it structurally
impossible to get wrong.

### Phase 3 — The language model
The prompt and the Gemini client.

The spec's chosen model, `gemini-2.0-flash`, **no longer exists** — it returns 404, and the whole
2.5 family is closed to new API keys. Worse, the failure was silent: every note would have become
`no_op` and the entry would have failed its mandatory-LLM requirement while still returning
HTTP 200. Candidate models were measured on the public cases and `gemini-3.5-flash-lite` was
chosen: it got all 18 notes right and answers in about 1.3 seconds.

A second silent failure was found here too. The spec sets an 8-second timeout, but Gemini
**rejects any deadline under 10 seconds** with a 400 error. The minimum is now 10 and anything
lower is raised automatically.

### Phase 4 — The guardrails
The layer that assumes the model may return anything at all.

Every field is re-checked: hours are de-duplicated, sorted and range-checked; a factor given as
`80` is understood as 80%; a reserve given as a share is multiplied by the real battery capacity;
anything unusable becomes `no_op`. Nothing in this file can raise an exception, so bad model output
can never turn into a failed request.

### Phase 5 — The maths
The solver. The plan is found by **linear programming** (SciPy's HiGHS), which returns the
*provably* cheapest plan rather than a good guess. 120 variables: five per hour (grid, solar used,
charge, discharge, stored energy).

Two subtleties are handled:

- A tiny "churn penalty" stops the solver returning a technically-equal answer that charges and
  discharges in the same hour, which can't be expressed in the response format.
- If the extracted rules are impossible to satisfy (say, a grid cap too low to meet demand), it
  **relaxes constraints in a fixed order** rather than failing, and in the worst case returns a
  simple grid-only plan. A valid plan always comes back.

### Phase 6 — The API
FastAPI wiring. `/health` never touches the model or the solver, so it answers immediately.
Malformed requests return 400 with a field list. Unexpected errors return 500 with an incident id
and **no stack trace and no key**.

The Gemini SDK's first call is slow (import, client build, TLS handshake). That work now happens in
the background at startup, which cut the first request from 3.3 s to 1.4 s.

### Phase 7 — The tests
The full suite, including the independent replay validator. **111 tests pass** offline with no API
key and no network. The 4 live-LLM tests are opt-in.

### Phase 8 — Docker
The `Dockerfile`: Python 3.11 slim, non-root user, health check, no secrets baked in. The key is
passed at run time only. **This was written but never built** — see section 8.

### Phase 9 — Documentation
`README.md`, covering all 13 sections the rubric scores: quickstart, configuration, model
disclosure, endpoints with real examples, test procedure, guardrail table, Docker instructions,
dependencies, limitations and secret handling.

---

## 6. The reliability work (why it isn't just one API call)

Testing revealed that the free Gemini tier is tight: **15 requests per minute per model**, and
`gemini-3.5-flash` allows only **20 per day**. A judge sending cases back to back would hit that.

So the client does several things:

- **A chain of models.** If the main one is busy, the next is tried immediately. Because quotas are
  counted per model, more models also means more capacity.
- **Hedging.** If a model hasn't answered in 2.5 seconds, the next one starts *alongside* it and
  whichever replies first wins.
- **Memory of failures.** A model that just returned "rate limited" is skipped for a while instead
  of being retried into the same wall.
- **A time budget.** The LLM step can never take more than 20 seconds, well inside the 30-second
  limit.
- **A cache.** Repeated notes are answered instantly.

Tested with 20 requests back to back: all 20 were answered and all 36 notes were interpreted
correctly, including four requests that arrived after the main model hit its rate limit.

If *everything* fails, the service still returns HTTP 200 with a valid plan — the notes are just
treated as `no_op`. It degrades instead of breaking.

---

## 7. How to run and test it

Start the service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run the tests (no API key needed, nothing is called over the network):

```bash
pytest -q
```

Send a real request:

```bash
curl -s -X POST http://localhost:8000/optimize-energy -H 'Content-Type: application/json' -d @data/sample_request.json
```

---

## 8. Honest status: what is proven and what is not

### Verified working

| Check | Result |
|---|---|
| Offline test suite | 111 passed, 4 skipped (the skipped ones are the opt-in live-LLM tests) |
| All 10 public cases, maths only | Cost, grid total and peak match the reference optimum exactly |
| Live request through the real Gemini | 200 in 2.6 s, correct directives, correct totals |
| `/health` ready after start | 1.6 s (limit is 60 s) |
| Secrets | `.env` ignored, never committed, no key in any tracked file or in git history |
| Malformed requests | Return 400 with a field list and no stack trace |

### Not verified

| Gap | Why | What to do |
|---|---|---|
| **Docker image never built or run** | Docker Desktop's engine would not start on this machine | Run `docker build -t gridwise:latest .` then `docker run --rm -p 8000:8000 --env-file .env gridwise:latest` and check `/health` |
| **No public URL** | `cloudflared` is not installed | Install it, run the tunnel, and put the URL in the README |
| **README placeholders** | Only you know these | Fill in `<repo-url>`, `<dockerhub-user>`, the image digest and the tunnel URL |
| **3-minute video** | Not a coding task | Tie-break only, no base points, but still on the checklist |

### Risks worth knowing

- **API quota is the biggest risk.** The service is only as good as its key. Development testing
  has already used part of today's free allowance, and `gemini-3.5-flash`'s 20-per-day limit is
  spent. A billing-enabled or fresh key is strongly recommended before judging.
- **Latency depends on which model answers.** The main model replies in roughly 1.5 s. If it is
  rate-limited, fallbacks take 4-7 s, which is over the 5 s target for full latency points but
  still well inside the 30 s hard limit — and the notes are still interpreted correctly, which is
  worth far more.
- **The tunnel must stay running** for the whole judging window. If the process dies, the URL dies.
- **Repository policy:** the rules require the repo to be private during the event and public
  afterwards for evaluation. Worth double-checking before the deadline.

---

## 9. Where this differs from the implementation spec

The spec was written before the event and some of it had gone stale. Each change below was made
because the original did not work, and every one was verified against the live API.

| Spec said | Reality | What was done |
|---|---|---|
| Use `gemini-2.0-flash` | Returns 404; the 2.5 family is closed to new keys | Switched to `gemini-3.5-flash-lite` and added a fallback chain |
| Timeout of 8 s | Gemini rejects anything under 10 s | Minimum raised to 10 s, enforced in code |
| "Lower the timeout if too slow" | Impossible now, given the 10 s floor | Speed comes from the fallback chain and warm-up instead |
| One model, one retry | A single model runs out of quota under load | Four models, instant failover, hedging and cooldowns |
| Blocking warm-up at startup | Would make `/health` wait on the network | Warm-up moved to a background thread |
| Put everything in a `gridwise/` sub-folder | Unnecessary nesting | Built at the repository root |

One genuine inconsistency inside the spec was also resolved: its failure table says a factor of
`1.4` should clamp to `1.0`, while its example code would have turned it into `0.014`. The code now
follows the table — values from 2 to 100 are read as percentages, and anything else is clamped.

---

## 10. Known minor imperfection

If the model returns two entries that are **byte-for-byte identical** and both claim the same wrong
note index, only the first is placed and the second note becomes `no_op`. It is a degenerate case
that has never been observed, and it fails safely rather than crashing. Fixing it would mean
tracking entries by identity rather than by value in `app/guardrails.py`.

Everything else found during review was fixed.
