"""Tools the agent can call, independent of Pipecat.

The voice bot (agent/bot.py) and the text eval agent (evals/text_agent.py) both
use these specs and ``run_tool``, so the evals exercise the same guardrails as
the live call.
"""

import asyncio
import json
import random
import re
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

from agent.borrowers import check_identity
from agent.call_state import CallState, Outcome
from agent.config import CONFIG, ROOT
from agent.history import recent_calls, summarise_for_agent
from agent.phrases import lender
from agent.speech import date_to_words, rupees_to_words, verbaliser

PAYMENTS_PATH = ROOT / "data" / "payments.json"

# --- Handlers ---------------------------------------------------------------------------


async def _verify_identity(state: CallState, args: dict) -> dict:
    if state.verified:
        return {"verified": True, "note": "Already verified."}
    if state.verification_locked:
        return {"verified": False, "note": "Too many attempts. Do not discuss the account. End the call politely."}

    state.verification_attempts += 1
    ok = check_identity(state.borrower, dob=args.get("dob"), phone_last4=args.get("phone_last4"))
    if not ok:
        remaining = max(0, 2 - state.verification_attempts)
        return {"verified": False, "attempts_remaining": remaining}

    state.verified = True
    b = state.borrower
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
    # Cross-call memory, released only now that the caller is verified.
    previous = await asyncio.to_thread(recent_calls, b.borrower_id, 3, state.history_db)
    if previous:
        result["previous_calls"] = summarise_for_agent(previous)
    return result


async def _send_payment_link(state: CallState, args: dict) -> dict:
    # Simulated provider latency and failures, to exercise filler speech and recovery.
    await asyncio.sleep(CONFIG.payment_link_delay_secs)
    if random.random() < CONFIG.payment_link_failure_rate:
        return {"sent": False, "error": "SMS provider timed out. Apologise and offer to try again or have the team resend it."}
    state.payment_link_sent = True
    return {"sent": True, "link_id": f"pl_{secrets.token_hex(4)}", "channel": "SMS to registered mobile"}


async def _check_payment_status(state: CallState, args: dict) -> dict:
    await asyncio.sleep(CONFIG.payment_check_delay_secs)  # mock lookup latency
    payments = json.loads(PAYMENTS_PATH.read_text()).get(state.borrower.borrower_id, [])[:3]
    return {
        "recent_payments": [
            {"date": p["date"], "date_spoken": date_to_words(date.fromisoformat(p["date"])),
             "amount_spoken": rupees_to_words(p["amount"]), "status": p["status"]}
            for p in payments
        ],
        "note": "Payments can take two to three working days to show. Never argue with the caller about this.",
    }


async def _end_call(state: CallState, args: dict) -> dict:
    record = state.finalize(
        Outcome(args["outcome"]),
        promise_to_pay_date=args.get("promise_to_pay_date"),
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
        "of their registered mobile number. Use it once they have confirmed their name and given a "
        "COMPLETE answer (day, month and year, or exactly four digits); if what you heard is "
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
        "promise_to_pay_date": {"type": "string", "description": "YYYY-MM-DD, if they committed to a date."},
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
