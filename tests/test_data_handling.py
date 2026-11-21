"""Audit log, retention and erasure."""

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from agent import audit, retention
from agent.borrowers import get_borrower
from agent.call_state import CallState
from agent.history import connect, record_call, record_verification_attempt
from agent.tools import run_tool

NOW = datetime(2026, 12, 31, 12, 0, tzinfo=UTC)
OLD_CALL, NEW_CALL = "20261101T100000000000", "20261230T100000000000"  # 60 days and 1 day before NOW


# --- Audit log ------------------------------------------------------------------------------------


def test_chain_verifies_and_details_are_redacted(tmp_path):
    log = tmp_path / "a.jsonl"
    audit.record("web:alice", "view_borrowers", path=log, count=8)
    audit.record("cli:sushant", "clear_history", "LF-1001", path=log, note="DOB 1990-03-14 given")
    assert audit.verify(log) == (True, 2, "OK: 2 entries, chain intact.")
    assert "1990" not in log.read_text()


def test_editing_a_line_breaks_the_chain(tmp_path):
    log = tmp_path / "a.jsonl"
    for i in range(3):
        audit.record("web:alice", "view_borrowers", path=log, n=i)
    lines = log.read_text().splitlines()
    edited = json.loads(lines[1])
    edited["actor"] = "web:someone-else"
    log.write_text("\n".join([lines[0], json.dumps(edited), lines[2]]) + "\n")
    ok, _, message = audit.verify(log)
    assert not ok and "Line 2 was modified" in message


def test_removing_a_line_breaks_the_chain(tmp_path):
    log = tmp_path / "a.jsonl"
    for i in range(3):
        audit.record("web:alice", "clear_history", "LF-1001", path=log, n=i)
    lines = log.read_text().splitlines()
    log.write_text("\n".join([lines[0], lines[2]]) + "\n")
    ok, _, message = audit.verify(log)
    assert not ok and "removed or reordered" in message


async def test_agent_disclosures_and_lockouts_are_audited(tmp_path):
    log = tmp_path / "agent.jsonl"
    state = CallState(borrower=get_borrower("LF-1002"), history_db=tmp_path / "h.db", audit_path=log)
    await run_tool(state, "verify_identity", {"dob": "2000-01-01"})  # a failed check releases nothing
    assert audit.entries(log) == []
    await run_tool(state, "verify_identity", {"phone_last4": "6789"})
    [entry] = audit.entries(log)
    assert (entry["actor"], entry["action"], entry["borrower_id"]) == ("agent", "disclose_loan_details", "LF-1002")
    assert entry["details"]["call_id"] == state.call_id


async def test_an_audit_failure_never_breaks_the_call(tmp_path):
    state = CallState(borrower=get_borrower("LF-1002"), history_db=tmp_path / "h.db", audit_path=tmp_path)  # a dir
    assert (await run_tool(state, "verify_identity", {"phone_last4": "6789"}))["verified"] is True


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from agent import web

    monkeypatch.setattr(web, "_tokens", lambda: {"tok-alice": "alice", "tok-bob": "bob"})
    return TestClient(web.app)


def test_web_access_is_audited_by_name(client):
    assert client.get("/api/borrowers", headers={"X-Admin-Token": "tok-alice"}).status_code == 200
    assert client.get("/borrowers?token=tok-bob").status_code == 200
    assert client.get("/api/borrowers", headers={"X-Admin-Token": "nope"}).status_code == 401
    assert [(e["actor"], e["action"]) for e in audit.entries()] == [("web:alice", "view_borrowers"),
                                                                   ("web:bob", "view_borrowers")]


def test_page_api_calls_reuse_the_openers_token(client):
    body = client.get("/borrowers?token=tok-bob").text
    assert 'const TOKEN = "tok-bob";' in body


# --- Retention ------------------------------------------------------------------------------------


@pytest.fixture
def outcomes(tmp_path):
    out = tmp_path / "outcomes"
    (out / "recordings").mkdir(parents=True)
    (out / "transcripts").mkdir()
    for cid in (OLD_CALL, NEW_CALL):
        (out / "recordings" / f"{cid}.wav").write_bytes(b"RIFF")
        (out / "transcripts" / f"{cid}.txt").write_text("AGENT: hello")
        (out / "transcripts" / f"{cid}.json").write_text(json.dumps({"outcome": {"borrower_id": "LF-1001"}}))
    (out / "bot.log").write_text("live log")
    rotated = out / "bot.2026-10-01_00-00-00_000000.log"
    rotated.write_text("old log")
    old = (NOW - timedelta(days=90)).timestamp()
    os.utime(rotated, (old, old))
    os.utime(out / "bot.log", (old, old))  # even an old-looking live log is never deleted
    return out


def test_purge_deletes_only_what_is_past_retention(outcomes, tmp_path):
    db, log = tmp_path / "h.db", tmp_path / "audit.jsonl"
    record_verification_attempt("LF-1001", "c", False, db, now=NOW - timedelta(days=45))
    record_verification_attempt("LF-1001", "c", False, db, now=NOW - timedelta(days=2))
    dry = retention.purge(30, now=NOW, outcomes=outcomes, db_path=db, dry_run=True, audit_path=log)
    assert len(dry["files"]) == 4 and (outcomes / "recordings" / f"{OLD_CALL}.wav").exists()
    assert audit.entries(log) == []

    done = retention.purge(30, now=NOW, outcomes=outcomes, db_path=db, audit_path=log)
    left = sorted(p.name for p in outcomes.rglob("*") if p.is_file())
    assert left == sorted(["bot.log", f"{NEW_CALL}.wav", f"{NEW_CALL}.txt", f"{NEW_CALL}.json"])
    assert done["verification_rows"] == 1
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM verification_attempts").fetchone()[0] == 1
    [entry] = audit.entries(log)
    assert entry["action"] == "retention_delete" and entry["details"]["recordings"] == 1


def test_retention_zero_keeps_everything(outcomes, tmp_path):
    assert retention.purge(0, now=NOW, outcomes=outcomes, db_path=tmp_path / "h.db")["disabled"]
    assert (outcomes / "recordings" / f"{OLD_CALL}.wav").exists()


def test_erase_removes_one_borrower_everywhere(outcomes, tmp_path):
    db, log = tmp_path / "h.db", tmp_path / "audit.jsonl"
    other = "20261229T100000000000"
    for cid, bid in ((OLD_CALL, "LF-1001"), (other, "LF-1002")):
        record_call({"call_id": cid, "borrower_id": bid, "outcome": "hardship", "verified": 1}, db)
    (outcomes / "recordings" / f"{other}.wav").write_bytes(b"RIFF")
    record_verification_attempt("LF-1001", OLD_CALL, False, db)

    result = retention.erase_borrower("LF-1001", "cli:test", outcomes=outcomes, db_path=db, audit_path=log)
    # NEW_CALL has no history row but its transcript names LF-1001, so it's found too.
    assert result == {"calls": 2, "files": 6, "history_rows": 1}
    assert sorted(p.name for p in (outcomes / "recordings").iterdir()) == [f"{other}.wav"]
    with connect(db) as conn:
        assert [r[0] for r in conn.execute("SELECT borrower_id FROM call_history")] == ["LF-1002"]
        assert conn.execute("SELECT COUNT(*) FROM verification_attempts").fetchone()[0] == 0
    assert audit.entries(log)[0]["action"] == "erase_borrower"


def test_erase_fails_closed_without_an_audit_entry(outcomes, tmp_path):
    with pytest.raises(OSError):
        retention.erase_borrower("LF-1001", "cli:test", outcomes=outcomes, db_path=tmp_path / "h.db",
                                 audit_path=tmp_path)  # a directory: the audit write fails
    assert (outcomes / "recordings" / f"{OLD_CALL}.wav").exists()  # nothing deleted
