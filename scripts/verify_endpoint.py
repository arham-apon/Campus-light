"""Run the ten public sample cases against a running GridWise service and grade the answers.

This is the same thing the judge harness does: POST each scenario, then re-check the returned plan
hour by hour against the official rules using tests/replay.py (which never imports the solver, so a
solver bug cannot hide behind a matching bug in the checker).

    python scripts/verify_endpoint.py                      # against http://localhost:8000
    python scripts/verify_endpoint.py <base-url>           # against the public tunnel URL

Exits 0 only if every case passes.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.replay import compile_expected, replay  # noqa: E402

TOL = 0.01
PACE_SECONDS = 2.0  # free-tier Gemini allows 15 requests/minute per model


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    pack = json.loads((Path(__file__).resolve().parents[1] / "data" /
                       "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json").read_text(encoding="utf-8"))

    print(f"Target: {base}\n")

    try:
        health = httpx.get(f"{base}/health", timeout=20)
        print(f"GET /health -> {health.status_code} {health.text.strip()}")
        if health.status_code != 200 or health.json().get("status") != "ok":
            print("FAIL: /health did not return {\"status\": \"ok\"}")
            return 1
    except Exception as exc:
        print(f"FAIL: cannot reach {base}/health ({type(exc).__name__})")
        return 1

    bad = httpx.post(f"{base}/optimize-energy", json={"scenario_id": "X"}, timeout=20)
    print(f"POST malformed body -> {bad.status_code} (expected 400)\n")

    failures: list[str] = []
    latencies: list[float] = []
    notes_ok = notes_total = 0

    for case in pack["cases"]:
        payload, expected = case["input"], case["expected_output"]
        started = time.perf_counter()
        try:
            response = httpx.post(f"{base}/optimize-energy", json=payload, timeout=40)
        except Exception as exc:
            failures.append(f"{case['id']}: request failed ({type(exc).__name__})")
            print(f"  {case['id']}  REQUEST FAILED {type(exc).__name__}")
            continue
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)

        if response.status_code != 200:
            failures.append(f"{case['id']}: HTTP {response.status_code}")
            print(f"  {case['id']}  HTTP {response.status_code}")
            continue
        body = response.json()
        problems: list[str] = []

        # 1. Interpretation: one entry per note, in order, matching organiser ground truth.
        truth = expected["directive_interpretation"]
        produced = body.get("directive_interpretation", [])
        if [e.get("note_index") for e in produced] != list(range(len(truth))):
            problems.append("note_index order")
        for got, want in zip(produced, truth):
            notes_total += 1
            if got.get("directive_type") != want["directive_type"] or got.get("applies") != want["applies"]:
                problems.append(f"note {want['note_index']}: {got.get('directive_type')} != {want['directive_type']}")
                continue
            adjustment, wanted = got.get("structured_adjustment"), want["structured_adjustment"]
            if wanted is None:
                if adjustment is not None:
                    problems.append(f"note {want['note_index']}: adjustment should be null")
                    continue
            else:
                if adjustment is None or set(adjustment) != set(wanted) or adjustment.get("hours") != wanted["hours"]:
                    problems.append(f"note {want['note_index']}: adjustment {adjustment} != {wanted}")
                    continue
                if any(abs(adjustment[k] - wanted[k]) > TOL for k in wanted if k != "hours"):
                    problems.append(f"note {want['note_index']}: numeric {adjustment} != {wanted}")
                    continue
            notes_ok += 1

        # 2. Plan validity: independent hour-by-hour replay against the true directives.
        base_solar = [h["solar_kwh"] for h in sorted(payload["hours"], key=lambda x: x["hour"])]
        problems += replay(body, payload, *compile_expected(truth, base_solar))

        # 3. Optimisation quality: must match the reference optimum.
        for field in ("total_cost_bdt", "total_grid_kwh", "peak_grid_kwh"):
            if abs(body.get(field, 1e9) - expected[field]) > TOL:
                problems.append(f"{field} {body.get(field)} != {expected[field]}")

        mark = "ok  " if not problems else "FAIL"
        print(f"  {case['id']}  {mark}  {elapsed:6.0f} ms  cost={body.get('total_cost_bdt')}")
        for problem in problems[:4]:
            print(f"        - {problem}")
        failures += [f"{case['id']}: {p}" for p in problems]
        time.sleep(PACE_SECONDS)

    ordered = sorted(latencies)
    print()
    if ordered:
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        print(f"latency: p50 {statistics.median(ordered):.0f} ms   p95 {p95:.0f} ms   max {ordered[-1]:.0f} ms"
              f"   (limit 30000 ms, target p95 <= 5000 ms)")
    print(f"notes interpreted correctly: {notes_ok}/{notes_total}")
    print(f"cases passed: {len(pack['cases']) - len({f.split(':')[0] for f in failures})}/{len(pack['cases'])}")
    print("\nRESULT:", "ALL CASES PASS" if not failures else f"{len(failures)} PROBLEM(S)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
