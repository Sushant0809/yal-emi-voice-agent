"""Cross-call memory: an SQLite log of every call's outcome.

Written once per call (on end_call, idle hang-up or disconnect) and read only
inside verify_identity, after verification succeeds. Previous calls are loan
information too, so they sit behind the same gate as the amount and due date.

Also keeps every identity check (pass or fail), so verification can lock out a
borrower across calls, not just within one.

    uv run python -m agent.history seed     # load data/call_history_seed.json (for demos)
    uv run python -m agent.history show LF-1006
    uv run python -m agent.history clear LF-1001   # forget one borrower's calls and failed checks (or `clear all`)
"""

import json
import sqlite3
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from agent import audit
from agent.config import CONFIG, ROOT
from agent.speech import date_to_words

SCHEMA = """
CREATE TABLE IF NOT EXISTS call_history (
    call_id             TEXT PRIMARY KEY,   -- idempotency key: a retried save is a no-op
    borrower_id         TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    verified            INTEGER NOT NULL,
    promise_to_pay_date TEXT,
    callback_time       TEXT,
    payment_link_sent   INTEGER NOT NULL DEFAULT 0,
    escalate_to_human   INTEGER NOT NULL DEFAULT 0,
    language            TEXT,
    notes               TEXT,
    duration_secs       REAL,
    turns               INTEGER,
    created_at          TEXT NOT NULL,
    prompt_version      TEXT
);
CREATE INDEX IF NOT EXISTS idx_call_history_borrower ON call_history (borrower_id, created_at DESC);
CREATE TABLE IF NOT EXISTS verification_attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    borrower_id TEXT NOT NULL,
    call_id     TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    created_at  TEXT NOT NULL              -- UTC ISO-8601, so string order is time order
);
CREATE INDEX IF NOT EXISTS idx_verification_borrower ON verification_attempts (borrower_id, created_at DESC);
"""

COLUMNS = [
    "call_id", "borrower_id", "outcome", "verified", "promise_to_pay_date", "callback_time",
    "payment_link_sent", "escalate_to_human", "language", "notes", "duration_secs", "turns",
    "created_at", "prompt_version",
]
# Columns added after the first release: (name, type), applied to existing databases on connect.
MIGRATIONS = [("prompt_version", "TEXT")]


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = Path(db_path or CONFIG.history_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(call_history)")}
    for name, kind in MIGRATIONS:
        if name not in have:
            conn.execute(f"ALTER TABLE call_history ADD COLUMN {name} {kind}")
    return conn


def record_call(row: dict, db_path: Path | None = None) -> None:
    """Insert one call. Saving the same call_id twice keeps the first row."""
    values = {c: row.get(c) for c in COLUMNS}
    values["created_at"] = values["created_at"] or datetime.now().isoformat(timespec="seconds")
    for flag in ("verified", "payment_link_sent", "escalate_to_human"):
        values[flag] = int(bool(values[flag]))
    with connect(db_path) as conn:
        # Only a duplicate call_id is ignored; any other constraint failure raises.
        # (INSERT OR IGNORE would also swallow NOT NULL violations and lose the row silently.)
        conn.execute(
            f"INSERT INTO call_history ({', '.join(COLUMNS)}) "
            f"VALUES ({', '.join('?' * len(COLUMNS))}) ON CONFLICT(call_id) DO NOTHING",
            [values[c] for c in COLUMNS],
        )


def recent_calls(borrower_id: str, limit: int = 3, db_path: Path | None = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM call_history WHERE borrower_id = ? ORDER BY created_at DESC LIMIT ?",
            (borrower_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def record_verification_attempt(borrower_id: str, call_id: str, ok: bool, db_path: Path | None = None,
                                now: datetime | None = None) -> None:
    now = now or datetime.now(UTC)
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO verification_attempts (borrower_id, call_id, ok, created_at) VALUES (?, ?, ?, ?)",
            (borrower_id, call_id, int(ok), now.isoformat(timespec="seconds")),
        )


def failed_verifications(borrower_id: str, hours: float, db_path: Path | None = None,
                         now: datetime | None = None) -> int:
    """Failed checks across all calls in the last `hours`, counting only those after the last success."""
    since = ((now or datetime.now(UTC)) - timedelta(hours=hours)).isoformat(timespec="seconds")
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(created_at) FROM verification_attempts WHERE borrower_id = ? AND ok = 1",
            (borrower_id,),
        ).fetchone()
        since = max(since, row[0]) if row[0] else since
        return conn.execute(
            "SELECT COUNT(*) FROM verification_attempts WHERE borrower_id = ? AND ok = 0 AND created_at > ?",
            (borrower_id, since),
        ).fetchone()[0]


def summarise_for_agent(calls: list[dict], today: date | None = None) -> list[dict]:
    """What the agent sees: compact, with dates already in spoken form."""
    today = today or date.today()
    summary = []
    for c in calls:
        called_on = datetime.fromisoformat(c["created_at"]).date()
        item = {
            "when": f"{(today - called_on).days} days ago ({date_to_words(called_on)})",
            "outcome": c["outcome"],
        }
        if promised := _parse_date(c.get("promise_to_pay_date")):  # a bad stored value is skipped
            item["promised_to_pay_on"] = date_to_words(promised)
            item["promise_date_has_passed"] = promised < today
        if c.get("callback_time"):
            item["asked_for_callback"] = c["callback_time"]
        if c.get("payment_link_sent"):
            item["payment_link_sent"] = True
        if c.get("notes"):
            item["notes"] = c["notes"]
        summary.append(item)
    return summary


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def seed(path: Path = ROOT / "data" / "call_history_seed.json", db_path: Path | None = None) -> int:
    from agent.dates import (
        resolve,  # the seed's dates are relative to today ("{-15}T11:20:00")
    )

    rows = resolve(json.loads(path.read_text()))
    for row in rows:
        record_call(row, db_path)
    return len(rows)


def clear_borrower(conn: sqlite3.Connection, borrower_id: str | None) -> int:
    """Forget a borrower's calls and identity checks (None = everyone). Returns calls removed."""
    where, args = ("WHERE borrower_id = ?", (borrower_id,)) if borrower_id else ("", ())
    conn.execute(f"DELETE FROM verification_attempts {where}", args)
    return conn.execute(f"DELETE FROM call_history {where}", args).rowcount


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "seed":
        print(f"Seeded {seed()} calls into {CONFIG.history_db}")
    elif cmd == "clear" and len(sys.argv) > 2:
        audit.record(audit.cli_actor(), "clear_history", None if sys.argv[2] == "all" else sys.argv[2],
                     scope=sys.argv[2])
        with connect() as conn:
            n = clear_borrower(conn, None if sys.argv[2] == "all" else sys.argv[2])
        print(f"Cleared {n} calls")
    elif cmd == "show" and len(sys.argv) > 2:
        audit.record(audit.cli_actor(), "view_history", sys.argv[2])
        for call in recent_calls(sys.argv[2], limit=20):
            print(json.dumps(call))
    else:
        print(__doc__)
