"""Rules enforced in code rather than left to the prompt."""

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from agent.config import CONFIG

IST = ZoneInfo("Asia/Kolkata")


def within_calling_hours(now: datetime | None = None, window: str | None = None) -> bool:
    """True if ``now`` falls inside the calling window ("08:00-19:00", borrower's time, IST).
    The end is exclusive: a call may not start at 19:00."""
    start_s, end_s = (window or CONFIG.calling_hours).split("-")
    local = (now or datetime.now(UTC)).astimezone(IST).time()
    return time.fromisoformat(start_s.strip()) <= local < time.fromisoformat(end_s.strip())
