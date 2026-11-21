"""Tools the agent can call, independent of Pipecat.

The voice bot (agent/bot.py) and the text eval agent (evals/text_agent.py) both
use these specs and ``run_tool``, so the evals exercise the same guardrails as
the live call.
"""

import asyncio
import hashlib
import json
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from loguru import logger

from agent import audit, dates
from agent.borrowers import check_identity
from agent.call_state import MAX_VERIFICATION_ATTEMPTS, CallState, Outcome
from agent.config import CONFIG, ROOT
from agent.history import (
    failed_verifications,
    recent_calls,
    record_verification_attempt,
    summarise_for_agent,
)
from agent.phrases import lender
from agent.speech import date_to_words, rupees_to_words, verbaliser

PAYMENTS_PATH = ROOT / "data" / "payments.json"

# --- Handlers ---------------------------------------------------------------------------


async def _verify_identity(state: CallState, args: dict) -> dict:
    if state.verified:
        return {"verified": True, "note": "Already verified."}
    if state.verification_locked:
        return {"verified": False, "note": "Too many attempts. Do not discuss the account. End the call politely."}

    dob = (args.get("dob") or "").strip()
    last4 = (args.get("phone_last4") or "").strip()
    # One factor per attempt: passing both would be two guesses for the price of one.
    if bool(dob) == bool(last4):
        what = "both a date of birth and mobile digits" if dob else "no date of birth or mobile digits"
        return {"verified": False, "note": f"Got {what}. Check exactly one of them. This did not count as an attempt."}

    b = state.borrower
    # Lockout across calls: someone guessing two DOBs per call, call after call.
    failed = await asyncio.to_thread(failed_verifications, b.borrower_id, CONFIG.verification_lockout_hours,
                                     state.history_db)
    if failed >= CONFIG.verification_lockout_failures:
        state.locked_out = True
        await _audit("verification_locked", state, failed_checks=failed)
        return {"verified": False, "note": "Too many failed attempts recently. Do not discuss the account. "
                                           "Say the team will be in touch, and end the call politely."}

    state.verification_attempts += 1
    ok = check_identity(b, dob=dob or None, phone_last4=last4 or None)
    await asyncio.to_thread(record_verification_attempt, b.borrower_id, state.call_id, ok, state.history_db)
    if not ok:
        # Whichever runs out first: this call's tries, or the borrower's budget across calls
        # (`failed` was counted before this attempt, so it's one higher now).
        this_call = MAX_VERIFICATION_ATTEMPTS - state.verification_attempts
        across_calls = CONFIG.verification_lockout_failures - (failed + 1)
        return {"verified": False, "attempts_remaining": max(0, min(this_call, across_calls))}

    details = b.loan_details()
    # Pre-rendered spoken forms so the model doesn't have to verbalise numbers itself.
    details["emi_amount_spoken"] = rupees_to_words(b.emi_amount)
    details["due_date_spoken"] = date_to_words(date.fromisoformat(b.due_date))
    if b.is_overdue:
        details["overdue_amount_spoken"] = rupees_to_words(b.overdue_amount)
    # Same, for a Hindi conversation.
    hi = verbaliser("hi")
    details["emi_amount_spoken_hi"] = hi.amount(b.emi_amount)
    details["due_date_spoken_hi"] = hi.date(date.fromisoformat(b.due_date))
    if b.is_overdue:
        details["overdue_amount_spoken_hi"] = hi.amount(b.overdue_amount)
    result = {"verified": True, "loan": details}
    # Cross-call memory, released only now that the caller is verified. Optional: a history
    # read failure must not cost a verified caller their reminder.
    try:
        previous = await asyncio.to_thread(recent_calls, b.borrower_id, 3, state.history_db)
        if previous:
            result["previous_calls"] = summarise_for_agent(previous)
    except Exception as e:
        logger.warning(f"Call history unavailable for {b.borrower_id}: {e}")
    # Only now: if building the result had failed, the caller would be "verified" with no
    # loan details in context, and every gated tool would open for a call that has stalled.
    state.verified = True
    await _audit("disclose_loan_details", state, previous_calls=len(result.get("previous_calls", [])))
    return result


async def _audit(action: str, state: CallState, **details) -> None:
    """The agent's own access to borrower data goes in the audit log too. Never breaks a call."""
    try:
        await asyncio.to_thread(audit.record, "agent", action, state.borrower.borrower_id,
                                path=state.audit_path, call_id=state.call_id, **details)
    except Exception as e:
        logger.warning(f"Audit write failed ({action}, call unaffected): {e}")


async def _send_payment_link(state: CallState, args: dict) -> dict:
    # Idempotent: one link per call, even if the model asks twice or calls it twice in one turn.
    async with state.link_lock:
        if state.payment_link_id:
            return {"sent": True, "already_sent": True, "link_id": state.payment_link_id,
                    "note": "Already sent on this call; don't send another. Tell them it's on its way."}
        # Simulated provider latency and failures, to exercise filler speech and recovery.
        await asyncio.sleep(CONFIG.payment_link_delay_secs)
        if random.random() < CONFIG.payment_link_failure_rate:
            return {"sent": False, "error": "SMS provider timed out. Apologise and offer to try again or have the team resend it."}
        # The call id is the idempotency key a real SMS provider would get, so a retried
        # request after a timeout can't send a second message.
        state.payment_link_id = "pl_" + hashlib.sha256(state.call_id.encode()).hexdigest()[:8]
        state.payment_link_sent = True
        return {"sent": True, "link_id": state.payment_link_id, "channel": "SMS to registered mobile"}


async def _check_payment_status(state: CallState, args: dict) -> dict:
    await asyncio.sleep(CONFIG.payment_check_delay_secs)  # mock lookup latency
    payments = json.loads(PAYMENTS_PATH.read_text()).get(state.borrower.borrower_id, [])[:3]
    paid_on = [dates.today() - timedelta(days=p["days_ago"]) for p in payments]  # mock dates are relative
    return {
        "recent_payments": [
            {"date": d.isoformat(), "date_spoken": date_to_words(d),
             "amount_spoken": rupees_to_words(p["amount"]), "status": p["status"]}
            for p, d in zip(payments, paid_on, strict=True)
        ],
        "note": "Payments can take two to three working days to show. Never argue with the caller about this.",
    }


def _promise_date(value: str | None, today: date) -> tuple[str | None, str]:
    """A promise-to-pay date the record can trust, or None plus the reason it can't be used.
    Stored dates are read back on later calls, so "next Friday" must never reach the database."""
    if not value:
        return None, ""
    try:
        promised = date.fromisoformat(value.strip())
    except ValueError:
        return None, f"{value!r} is not a calendar date."
    if promised < today:
        return None, f"{value} is in the past (today is {today.isoformat()})."
    return promised.isoformat(), ""


async def _end_call(state: CallState, args: dict) -> dict:
    try:
        outcome = Outcome(args.get("outcome"))
    except ValueError:
        return {"error": f"Unknown outcome {args.get('outcome')!r}. Call this again with one from the list."}
    promise, problem = _promise_date(args.get("promise_to_pay_date"), date.today())
    if problem:  # nothing recorded and the call stays open, like an unknown outcome
        return {"error": f"Not recorded: promise_to_pay_date {problem} Confirm the exact date with them "
                         "(day and month), then call this again with it as YYYY-MM-DD."}
    record = state.finalize(
        outcome,
        promise_to_pay_date=promise,
        callback_time=args.get("callback_time"),
        escalate_to_human=bool(args.get("escalate_to_human")),
        notes=args.get("notes", ""),
    )
    return {"recorded": True, "outcome": record.outcome.value}


# --- Registry -----------------------------------------------------------------------------
# A tool is one declaration: what it does and when to use it (its description, which the
# model reads on every request), its parameters, its handler, and the rules the runtime
# enforces for it. Adding a tool means adding an entry here; the system prompt never names
# tools, and the bot and eval agent pick up every registered tool automatically.


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    properties: dict
    required: list[str]
    handler: Callable[[CallState, dict], Awaitable[dict]]
    requires_verification: bool = False  # refused by run_tool until the caller is verified
    filler: str | None = None  # phrase-pack key spoken while it runs (dead-air cover)
    ends_call: bool = False  # the runtime speaks its `goodbye` and hangs up after it


REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    REGISTRY[tool.name] = tool
    return tool


register(Tool(
    name="verify_identity",
    description=(
        "Check that the person is the borrower, using their date of birth or the last four digits "
        "of their registered mobile number: pass exactly ONE of the two per check. Use it once they "
        "have confirmed their name and given a COMPLETE answer (day, month and year, or exactly "
        "four digits); if what you heard is "
        "incomplete, ask for the missing part first, because a failed check uses up one of only two "
        "attempts. Don't re-check details that already failed. On success it returns the loan "
        "details (with amounts and dates already in spoken form) and any previous calls; nothing "
        "about the loan is available before this succeeds."
    ),
    properties={
        "dob": {"type": "string", "description": "Date of birth as YYYY-MM-DD, if given."},
        "phone_last4": {"type": "string", "description": "Last four digits of the mobile number, if given."},
    },
    required=[],
    handler=_verify_identity,
))

register(Tool(
    name="send_payment_link",
    description=(
        "Send a secure payment link by SMS to the borrower's registered mobile. Offer it when they "
        "plan to pay and haven't yet, and only send it once they agree. Takes a few seconds; may "
        "fail, in which case apologise and offer to try again or have the team resend it."
    ),
    properties={},
    required=[],
    handler=_send_payment_link,
    requires_verification=True,
    filler="fillers.send_payment_link",
))

register(Tool(
    name="check_payment_status",
    description=(
        "Look up the borrower's most recent payments (date, amount, status). Useful when they say "
        "they've already paid. If you can't see their payment, say records can take a few days to "
        "update and offer to have the team check; never argue or suggest they're not telling the truth."
    ),
    properties={},
    required=[],
    handler=_check_payment_status,
    requires_verification=True,
    filler="fillers.check_payment_status",
))

register(Tool(
    name="end_call",
    description=(
        "End the call and record its outcome. Use it once, when the call is done: their plan is "
        "known and anything agreed is done, they asked to stop, it's the wrong person, or "
        "verification failed twice. Never use it while they're waiting on you: if their last "
        "turn was a question or a concern (\"is it not showing?\"), answer it first and let them "
        "respond. If they say they need to go once you know their plan, use it in that same "
        "turn. Put your closing words in `goodbye`: they are spoken for you right before hanging "
        "up, so don't also write them as text."
    ),
    properties={
        "goodbye": {
            "type": "string",
            "description": (
                "EVERYTHING you still need to say before hanging up, in the call's language; text "
                "written next to this tool call may not be spoken. First answer their last question or "
                "concern if they raised one (e.g. what you found when you checked their payment), then "
                "confirm anything done or agreed (link sent, date noted, specialist will call), then "
                "thanks and goodbye. E.g. 'I can see your payment from the seventeenth is still "
                "processing, so you're all set. Sorry for the bother. Thank you, Ananya. Goodbye.'"
            ),
        },
        "outcome": {"type": "string", "enum": [o.value for o in Outcome if o is not Outcome.DISCONNECTED],
                    "description": "What happened on THIS call (never a copy of an earlier call's outcome)."},
        "promise_to_pay_date": {"type": "string", "description": (
            "YYYY-MM-DD, if they committed to a date. Must be a real date, today or later; otherwise "
            "nothing is recorded and you get an error, so confirm the date with them first.")},
        "callback_time": {"type": "string", "description": "When to call back, if they asked for a callback."},
        "escalate_to_human": {"type": "boolean", "description": "True for disputes, hardship, or if they asked for a person."},
        "notes": {"type": "string", "description": "One sentence a human agent would find useful."},
    },
    required=["goodbye", "outcome", "escalate_to_human", "notes"],
    handler=_end_call,
    ends_call=True,
))

def tool_specs() -> list[dict[str, Any]]:
    """Provider-agnostic specs for every registered tool (Pipecat and the eval agent build
    theirs from these). A function, so tools registered later are picked up too."""
    return [
        {"name": t.name, "description": t.description, "properties": t.properties, "required": t.required}
        for t in REGISTRY.values()
    ]


async def run_tool(state: CallState, name: str, args: dict) -> dict:
    tool = REGISTRY.get(name)
    if not tool:
        return {"error": f"Unknown tool {name}"}
    # One gate for every tool, current and future: account actions need a verified caller.
    if tool.requires_verification and not state.verified:
        return {"error": "Not available until the caller's identity is verified. Don't share or do anything account-related yet."}
    try:
        if state.inject_failures.get(name, 0) > 0:  # evals only: exercise the real error path
            state.inject_failures[name] -= 1
            raise TimeoutError("provider timed out")
        return await tool.handler(state, args)
    except Exception as e:  # a failing tool must never crash the call
        return {"error": f"Tool failed: {e}. Apologise briefly and continue."}


_GOODBYE_WORDS = re.compile(r"bye|thank|take care|धन्यवाद|नमस्ते|शुक्रिया", re.I)


def closing_line(args: dict, spoken_this_turn: str, language: str) -> str | None:
    """What to say before hanging up, or None if the agent already said its goodbye as text.

    Why this exists: on Sonnet 5.5, text written right before a tool call can come back as a
    (hidden) progress-update thinking block instead of a text block. Seen in evals: after
    send_payment_link the reply was [thinking, tool_use(end_call)], so the spoken "link sent,
    thank you, goodbye" never reached TTS. Carrying the goodbye in the tool call avoids that.
    """
    # The agent already spoke this turn: speaking the field too would overlap with its own
    # closing (seen live: two closings interleaved word by word in token-streamed TTS).
    if _GOODBYE_WORDS.search(spoken_this_turn or "") or len((spoken_this_turn or "").split()) >= 4:
        return None
    goodbye = (args.get("goodbye") or "").strip()
    # Last resort (goodbye is a required field): never hang up in silence.
    return goodbye or lender().phrase(language, "fallback_goodbye")
