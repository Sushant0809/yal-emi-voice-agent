"""Mock borrower records and identity verification."""

import json
import re
from dataclasses import dataclass
from datetime import date
from functools import cache

from agent.config import CONFIG


@dataclass(frozen=True)
class Borrower:
    borrower_id: str
    name: str
    phone: str
    dob: str
    phone_last4: str
    loan_type: str
    emi_amount: int
    due_date: str
    overdue_amount: int  # unpaid amount past its due date; 0 when the borrower is up to date
    preferred_language: str

    @property
    def first_name(self) -> str:
        return self.name.split()[0]

    @property
    def is_overdue(self) -> bool:
        return self.overdue_amount > 0

    @property
    def days_overdue(self) -> int:
        """Days since the due date, counted from today. It used to be a fixed number in the data,
        which drifted: on 21 Nov the agent still said "eight days overdue" for a 10 Nov due date."""
        if not self.is_overdue:
            return 0
        return max(0, (_today() - date.fromisoformat(self.due_date)).days)

    def loan_details(self) -> dict:
        """Details the agent may speak only after verification."""
        return {
            "loan_type": self.loan_type,
            "emi_amount_inr": self.emi_amount,
            "due_date": self.due_date,
            "days_overdue": self.days_overdue,
            "overdue_amount_inr": self.overdue_amount,
        }


def _today() -> date:
    """Today's date (a function, so tests can pin it)."""
    return date.today()


@cache
def load_borrowers() -> dict[str, Borrower]:
    records = json.loads(CONFIG.borrowers_path.read_text())
    return {r["borrower_id"]: Borrower(**r) for r in records}


def get_borrower(borrower_id: str) -> Borrower:
    return load_borrowers()[borrower_id]


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def check_identity(borrower: Borrower, *, dob: str | None = None, phone_last4: str | None = None) -> bool:
    """True if either verification factor matches.

    The LLM normalises what the caller said into ``YYYY-MM-DD`` / four digits before
    calling the tool; we still compare loosely on digits so "14/03/1990" also works.
    """
    if phone_last4 and _digits(phone_last4)[-4:] == borrower.phone_last4:
        return True
    if dob:
        given = _digits(dob)
        expected = date.fromisoformat(borrower.dob)
        candidates = {
            expected.strftime("%Y%m%d"),
            expected.strftime("%d%m%Y"),
            expected.strftime("%m%d%Y"),
        }
        return given in candidates
    return False
