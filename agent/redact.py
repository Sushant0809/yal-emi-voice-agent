"""Mask personal data before it reaches a log file or a trace.

Two layers: values under known-sensitive keys are replaced outright (tool arguments such
as ``dob``), and free text (transcripts, Pipecat's debug lines) has dates of birth, phone
numbers and four-digit groups masked, borrower names replaced, and loan amounts and due dates
(in digits or words, English or Hindi) masked. Call ids and most amounts
survive (a four-digit amount like 4850 is masked too; that's the price of catching every
"last four").
"""

import re
from functools import cache
from typing import Any

SENSITIVE_KEYS = {"dob", "phone_last4", "phone", "name", "full_name", "first_name",
                  # Loan information: the verify_identity and check_payment_status results.
                  "loan", "previous_calls", "recent_payments"}
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


def _words_pattern(words: str) -> str:
    """Spoken number or date as a regex that tolerates the ways it gets written: hyphens or
    spaces, an optional "and" ("four thousand eight hundred and fifty"), and Hindi written
    with or without nukta and chandrabindu."""
    parts = [re.escape(w) for w in re.split(r"[\s-]+", words.strip()) if w]
    body = r"[\s-]+(?:and[\s-]+)?".join(parts)
    return body.replace("\u093c", "\u093c?").replace("\u0901", "[\u0901\u0902]")


@cache
def _loan_values() -> tuple[re.Pattern, re.Pattern] | None:
    """Every borrower's loan amounts and due dates, in digits, English words and Hindi words.
    Pipecat logs the whole LLM context and every TTS line at DEBUG, and those carry the amount
    and due date in words after verification. Mock data makes the full list cheap; a production
    system would scrub with the borrower record of each call, as for names."""
    try:
        from datetime import date

        from agent.borrowers import load_borrowers
        from agent.speech import (
            date_to_words,
            hindi_number_to_words,
            number_to_words,
            verbaliser,
        )

        amounts, dates = set(), set()
        for b in load_borrowers().values():
            for a in {b.emi_amount, b.overdue_amount} - {0}:
                amounts.add(",?".join(str(a)))  # 12500 and 12,500
                amounts.add(_words_pattern(number_to_words(a)))
                amounts.add(_words_pattern(hindi_number_to_words(a)))
            d = date.fromisoformat(b.due_date)
            month = d.strftime("%B")
            dates.add(_words_pattern(date_to_words(d).removeprefix("the ")))  # "fifth of December"
            dates.add(_words_pattern(verbaliser("hi").date(d)))
            dates.add(rf"{d.day}(?:st|nd|rd|th)?\s+(?:of\s+)?{month}|{month}\s+{d.day}(?:st|nd|rd|th)?")
    except Exception:  # redaction must never fail because the borrower file can't be read
        return None
    longest_first = lambda ps: "|".join(sorted(ps, key=len, reverse=True))  # noqa: E731
    return (re.compile(rf"(?<![\w,])(?:{longest_first(amounts)})(?![\w,])", re.I),
            re.compile(rf"(?<!\w)(?:{longest_first(dates)})(?!\w)", re.I))


def redact_text(text: str) -> str:
    if loan := _loan_values():  # before the digit patterns, so "4,850" is masked whole
        amounts, dates = loan
        text = dates.sub("[date]", amounts.sub("[amount]", text))
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
