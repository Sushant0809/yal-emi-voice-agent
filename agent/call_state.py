"""Per-call state and the structured outcome record.

The state lives in code, not in the LLM. The LLM decides what to say; this object
decides what it is *allowed* to know (loan details only after verification) and
what gets recorded.
"""

import asyncio
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, PrivateAttr

from agent.borrowers import Borrower
from agent.config import CONFIG
from agent.history import record_call

MAX_VERIFICATION_ATTEMPTS = 2


class Outcome(StrEnum):
    WILL_PAY_ON_TIME = "will_pay_on_time"
    ALREADY_PAID = "already_paid"
    PROMISE_TO_PAY = "promise_to_pay"
    HARDSHIP = "hardship"
    DISPUTE = "dispute"
    WRONG_NUMBER = "wrong_number"
    THIRD_PARTY = "third_party_answered"
    CALLBACK_REQUESTED = "callback_requested"
    VERIFICATION_FAILED = "verification_failed"
    REFUSED_TO_ENGAGE = "refused_to_engage"
    NO_RESPONSE = "no_response"
    DISCONNECTED = "disconnected"


# Outcomes that always need a human to follow up, whatever the LLM says.
ALWAYS_ESCALATE = {Outcome.HARDSHIP, Outcome.DISPUTE}


class OutcomeRecord(BaseModel):
    borrower_id: str
    verified: bool
    outcome: Outcome
    promise_to_pay_date: str | None = None
    callback_time: str | None = None
    payment_link_sent: bool = False
    escalate_to_human: bool = False
    language: str = "en"
    notes: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_secs: float = 0.0
    turns: int = 0
    connection_incidents: int = 0  # provider drop-outs the agent recovered from
    prompt_version: str = ""  # agent/prompts.py PROMPT_VERSION: which wording produced this call


class CallState(BaseModel):
    borrower: Borrower
    call_id: str = Field(default_factory=lambda: datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
    verified: bool = False
    verification_attempts: int = 0
    locked_out: bool = False  # too many failed checks across calls (see history.failed_verifications)
    payment_link_sent: bool = False
    payment_link_id: str | None = None
    language: str = "en"
    turns: int = 0
    idle_prompts: int = 0
    connection_incidents: int = 0
    started_monotonic: float = Field(default_factory=time.monotonic)
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    outcome: OutcomeRecord | None = None
    # Where cross-call memory lives. Evals point this at a throwaway database per run.
    history_db: Path = CONFIG.history_db
    # Where the agent's own data access is audited (None = the real log). Evals use a throwaway one.
    audit_path: Path | None = None
    # Fault injection for evals: tool name -> how many of its next calls should fail.
    inject_failures: dict[str, int] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
    # Serialises send_payment_link, so two tool calls in one turn can't send two links.
    _link_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @property
    def link_lock(self) -> asyncio.Lock:
        return self._link_lock

    def record_incident(self) -> None:
        self.connection_incidents += 1

    @property
    def ended(self) -> bool:
        return self.outcome is not None

    @property
    def verification_locked(self) -> bool:
        return not self.verified and (self.locked_out or self.verification_attempts >= MAX_VERIFICATION_ATTEMPTS)

    def finalize(
        self,
        outcome: Outcome,
        *,
        promise_to_pay_date: str | None = None,
        callback_time: str | None = None,
        escalate_to_human: bool = False,
        notes: str = "",
    ) -> OutcomeRecord:
        """Record the outcome once; later calls return the first record unchanged."""
        if self.outcome:
            return self.outcome
        from agent.prompts import prompt_version  # lazy: prompts -> tools -> call_state

        self.outcome = OutcomeRecord(
            borrower_id=self.borrower.borrower_id,
            verified=self.verified,
            outcome=outcome,
            promise_to_pay_date=promise_to_pay_date,
            callback_time=callback_time,
            payment_link_sent=self.payment_link_sent,
            escalate_to_human=escalate_to_human or outcome in ALWAYS_ESCALATE,
            language=self.language,
            notes=notes,
            started_at=self.started_at,
            ended_at=datetime.now(UTC).isoformat(),
            duration_secs=round(time.monotonic() - self.started_monotonic, 2),
            turns=self.turns,
            connection_incidents=self.connection_incidents,
            prompt_version=prompt_version(),
        )
        return self.outcome

    def save(self) -> None:
        """Persist the outcome to call history. Safe to call more than once."""
        if not self.outcome:
            return
        record_call({"call_id": self.call_id, **self.outcome.model_dump(mode="json")}, self.history_db)
