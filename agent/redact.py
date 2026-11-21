"""Mask personal data before it reaches a log file or a trace.

Two layers: values under known-sensitive keys are replaced outright (tool arguments such
as ``dob``), and free text (transcripts, Pipecat's debug lines) has dates of birth, phone
numbers and four-digit groups masked, and borrower names replaced. Call ids and most amounts
survive (a four-digit amount like 4850 is masked too; that's the price of catching every
"last four").
"""

import re
from functools import cache
from typing import Any

SENSITIVE_KEYS = {"dob", "phone_last4", "phone", "name", "full_name", "first_name"}
MASK = "[redacted]"

_NUM_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_PATTERNS = [
    re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b"),          # 1990-03-14
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"),        # 14/03/1990
    re.compile(r"\+?\b\d[\d -]{8,}\d\b"),                      # phone numbers
    re.compile(r"\b\d{4}\b"),                                  # last four digits, birth years
    re.compile(rf"\b{_NUM_WORD}(?:[\s,-]+{_NUM_WORD}){{3,}}\b", re.I),  # "two three four five"
]


@cache
def _names() -> re.Pattern | None:
    """Every borrower's full and partial name (Latin script). Names in Devanagari are not
    caught; a production system would scrub with the borrower record of each call."""
    try:
        from agent.borrowers import load_borrowers

        parts = {p for b in load_borrowers().values() for p in (b.name, *b.name.split()) if len(p) >= 3}
    except Exception:  # redaction must never fail because the borrower file can't be read
        return None
    return re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, parts), key=len, reverse=True)) + r")\b")


def redact_text(text: str) -> str:
    for pattern in _PATTERNS:
        text = pattern.sub(MASK, text)
    if names := _names():
        text = names.sub("[name]", text)
    return text


def redact(value: Any) -> Any:
    """A copy of ``value`` with sensitive keys masked and free text scrubbed (dicts, lists, str)."""
    if isinstance(value, dict):
        return {k: (MASK if k in SENSITIVE_KEYS and v else redact(v)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_log_record(record: dict) -> None:
    """Loguru patcher: scrub every log line before any sink (console or bot.log) writes it."""
    record["message"] = redact_text(record["message"])
