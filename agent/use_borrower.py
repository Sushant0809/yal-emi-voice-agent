"""Pick which borrower the next browser call goes to (no restart needed).

    uv run python -m agent.use_borrower            # list borrowers
    uv run python -m agent.use_borrower LF-1003    # next call goes to Amit Verma
"""

import re
import sys

from agent import audit
from agent.borrowers import load_borrowers
from agent.config import ROOT

ENV = ROOT / ".env"


def set_borrower(bid: str) -> None:
    """Persist the choice in .env; the bot re-reads it at the start of every call."""
    text = ENV.read_text()
    line = f"DEFAULT_BORROWER_ID={bid}"
    if re.search(r"^DEFAULT_BORROWER_ID=", text, flags=re.M):
        text = re.sub(r"^DEFAULT_BORROWER_ID=.*$", line, text, flags=re.M)
    else:
        text += f"\n{line}\n"
    ENV.write_text(text)


def main() -> None:
    borrowers = load_borrowers()
    if len(sys.argv) < 2:
        audit.record(audit.cli_actor(), "view_borrowers", count=len(borrowers))  # prints DOBs
        for b in borrowers.values():
            status = f"{b.days_overdue} days overdue" if b.is_overdue else f"due {b.due_date}"
            print(f"{b.borrower_id}  {b.name:16s} {b.loan_type:22s} {b.preferred_language}  "
                  f"DOB {b.dob}  mobile ...{b.phone_last4}  {status}")
        return
    bid = sys.argv[1].upper()
    if bid not in borrowers:
        sys.exit(f"Unknown borrower {bid}. Options: {', '.join(borrowers)}")
    audit.record(audit.cli_actor(), "select_borrower", bid)
    set_borrower(bid)
    b = borrowers[bid]
    print(f"Next call -> {b.name} ({bid}). Verify with DOB {b.dob} or mobile ...{b.phone_last4}.")


if __name__ == "__main__":
    main()
