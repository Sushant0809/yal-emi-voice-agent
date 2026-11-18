"""Cross-call memory: an SQLite log of every call's outcome.

Written once per call (on end_call, idle hang-up or disconnect) and read only
inside verify_identity, after verification succeeds. Previous calls are loan
information too, so they sit behind the same gate as the amount and due date.

    uv run python -m agent.history seed     # load data/call_history_seed.json (for demos)
    uv run python -m agent.history show LF-1006
    uv run python -m agent.history clear LF-1001   # forget one borrower's calls (or `clear all`)
"""

import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

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
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_call_history_borrower ON call_history (borrower_id, created_at DESC);
"""

COLUMNS = [
    "call_id", "borrower_id", "outcome", "verified", "promise_to_pay_date", "callback_time",
    "payment_link_sent", "escalate_to_human", "language", "notes", "duration_secs", "turns",
    "created_at",
]


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = Path(db_path or CONFIG.history_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
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
        if c.get("promise_to_pay_date"):
            promised = date.fromisoformat(c["promise_to_pay_date"])
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


def seed(path: Path = ROOT / "data" / "call_history_seed.json", db_path: Path | None = None) -> int:
    rows = json.loads(path.read_text())
    for row in rows:
        record_call(row, db_path)
    return len(rows)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "seed":
        print(f"Seeded {seed()} calls into {CONFIG.history_db}")
    elif cmd == "clear" and len(sys.argv) > 2:
        with connect() as conn:
            if sys.argv[2] == "all":
                n = conn.execute("DELETE FROM call_history").rowcount
            else:
                n = conn.execute("DELETE FROM call_history WHERE borrower_id = ?", (sys.argv[2],)).rowcount
        print(f"Cleared {n} calls")
    elif cmd == "show" and len(sys.argv) > 2:
        for call in recent_calls(sys.argv[2], limit=20):
            print(json.dumps(call))
    else:
        print(__doc__)
