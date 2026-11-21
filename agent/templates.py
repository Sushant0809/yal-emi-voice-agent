"""Templated turns: fixed words spoken without waiting for the LLM.

The LLM's time to first token (~1.0-1.2s) is the largest part of every reply. Two turns
are predictable on every call and are spoken from the lender's phrase pack instead
(config/lenders/<lender>/<lang>.yaml), filled from the borrower record:

- the greeting, as soon as the call connects;
- the reminder right after a successful verification, which otherwise needs a second
  LLM pass after the tool result.

The LLM still writes every turn that needs judgement, including the reminder when there
are previous calls (a broken promise or pending dispute must be raised sensitively).
Both the live bot and the eval agent use these, so the evals test the same behaviour.
"""

from datetime import date

from agent.borrowers import Borrower
from agent.config import CONFIG
from agent.phrases import lender
from agent.speech import verbaliser

# An instruction to the LLM, not spoken, so it stays in English.
CALL_CONNECTED_NOTE = (
    "[The call has just connected. You have already greeted the person and asked if you are "
    "speaking with {name}. Wait for their answer.]"
)


def greeting(borrower: Borrower, recorded: bool | None = None) -> str:
    """The opening line. When the call is recorded it says so, before anything else is asked."""
    L, lang = lender(), borrower.preferred_language
    recorded = CONFIG.record_calls if recorded is None else recorded
    notice = " " + L.phrase(lang, "recording_notice") if recorded else ""
    return L.phrase(lang, "greeting", name=borrower.name, first_name=borrower.first_name, recording_notice=notice)


def outside_hours(borrower: Borrower) -> str:
    """Said instead of the greeting outside calling hours, before hanging up. No loan details."""
    return lender().phrase(borrower.preferred_language, "outside_hours")


def reminder(borrower: Borrower, verify_result: dict, language: str) -> str | None:
    """The post-verification reminder, or None when the LLM should write it instead."""
    if not verify_result.get("verified") or verify_result.get("previous_calls"):
        return None
    b, v, L = borrower, verbaliser(language), lender()
    values = {
        "first_name": b.first_name,
        "loan_type": L.loan_type(language, b.loan_type),
        "due_date": v.date(date.fromisoformat(b.due_date)),
    }
    if b.is_overdue:
        return L.phrase(language, "reminder_overdue", amount=v.amount(b.overdue_amount),
                        days_overdue=v.days(b.days_overdue), **values)
    return L.phrase(language, "reminder_due", amount=v.amount(b.emi_amount), **values)
