"""Dates in the mock world, relative to today.

The mock data (borrowers, payments, seeded call history) and the eval scenarios describe a
world anchored to the day they're used: Priya is always 8 days overdue, Rahul's EMI is always
due in 17 days, a promise in a scenario is always next week. Fixed dates went stale: an agent
saying "eight days overdue" for a due date 11 days ago, and scenario promise dates that had
already passed.

Strings may contain date tokens, filled in by ``resolve``:

    {+7}          2026-11-25                (ISO; also "{-8}T11:20:00" for timestamps)
    {+7:th}       25th November
    {+7:day_th}   25th
    {+7:spoken}   the twenty-fifth of November
    {+7:ordinal}  twenty-fifth
    {+7:hi}       पच्चीस नवंबर
    {+7:hi_day}   पच्चीस
"""

import re
from datetime import date, timedelta
from typing import Any

TOKEN = re.compile(r"\{([+-]\d+)(?::(\w+))?\}")


def today() -> date:
    """The anchor date. A function, so tests can pin it."""
    return date.today()


def _suffix(day: int) -> str:
    return "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def render(d: date, fmt: str | None) -> str:
    from agent.speech import date_to_words, hindi_number_to_words, verbaliser

    match fmt:
        case None | "iso":
            return d.isoformat()
        case "th":
            return f"{d.day}{_suffix(d.day)} {d:%B}"
        case "day_th":
            return f"{d.day}{_suffix(d.day)}"
        case "spoken":
            return date_to_words(d)
        case "ordinal":
            return date_to_words(d).split()[1]
        case "hi":
            return verbaliser("hi").date(d)
        case "hi_day":
            return hindi_number_to_words(d.day)
    raise ValueError(f"Unknown date format {fmt!r}")


def resolve(value: Any, anchor: date | None = None) -> Any:
    """``value`` with every date token filled in (strings, and inside dicts and lists)."""
    anchor = anchor or today()
    if isinstance(value, str):
        return TOKEN.sub(lambda m: render(anchor + timedelta(days=int(m.group(1))), m.group(2)), value)
    if isinstance(value, dict):
        return {k: resolve(v, anchor) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, anchor) for v in value]
    return value
