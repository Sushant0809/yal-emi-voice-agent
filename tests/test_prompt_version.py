"""Prompt changes must be deliberate: the wording is pinned to PROMPT_VERSION."""

import json

from agent.borrowers import get_borrower
from agent.call_state import CallState, Outcome
from agent.history import recent_calls
from agent.prompts import PROMPT_VERSION, VERSIONS_PATH, content_hash, prompt_version


def test_prompt_wording_matches_the_current_version():
    known = json.loads(VERSIONS_PATH.read_text())
    assert PROMPT_VERSION in known, f"Add {PROMPT_VERSION} to {VERSIONS_PATH.name}"
    assert known[PROMPT_VERSION]["hash"] == content_hash(), (
        f"The prompt or a tool description changed. Bump PROMPT_VERSION in agent/prompts.py and add "
        f'"<new version>": {{"hash": "{content_hash()}", "notes": "<what changed>"}} to {VERSIONS_PATH.name}.'
    )
    assert prompt_version() == PROMPT_VERSION


def test_every_version_has_a_note_and_a_unique_hash():
    known = json.loads(VERSIONS_PATH.read_text())
    assert all(v.get("notes") for v in known.values())
    hashes = [v["hash"] for v in known.values()]
    assert len(hashes) == len(set(hashes))


def test_outcome_record_and_history_carry_the_prompt_version(tmp_path):
    state = CallState(borrower=get_borrower("LF-1001"), history_db=tmp_path / "h.db")
    state.finalize(Outcome.WILL_PAY_ON_TIME)
    state.save()
    assert state.outcome.prompt_version == PROMPT_VERSION
    assert recent_calls("LF-1001", db_path=tmp_path / "h.db")[0]["prompt_version"] == PROMPT_VERSION


def test_old_databases_gain_the_column(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:  # the schema before prompt_version existed
        conn.execute("CREATE TABLE call_history (call_id TEXT PRIMARY KEY, borrower_id TEXT NOT NULL, "
                     "outcome TEXT NOT NULL, verified INTEGER NOT NULL, promise_to_pay_date TEXT, "
                     "callback_time TEXT, payment_link_sent INTEGER NOT NULL DEFAULT 0, "
                     "escalate_to_human INTEGER NOT NULL DEFAULT 0, language TEXT, notes TEXT, "
                     "duration_secs REAL, turns INTEGER, created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO call_history (call_id, borrower_id, outcome, verified, created_at) "
                     "VALUES ('old', 'LF-1001', 'hardship', 1, '2026-11-01T10:00:00')")
    state = CallState(borrower=get_borrower("LF-1001"), history_db=db)
    state.finalize(Outcome.WILL_PAY_ON_TIME)
    state.save()
    rows = {r["call_id"]: r for r in recent_calls("LF-1001", db_path=db)}
    assert rows["old"]["prompt_version"] is None and rows[state.call_id]["prompt_version"] == PROMPT_VERSION
