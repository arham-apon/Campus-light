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
