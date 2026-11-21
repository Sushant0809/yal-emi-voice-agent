"""Phase 1 hardening: each test pins one fix."""

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest

from agent.borrowers import get_borrower
from agent.call_state import CallState
from agent.compliance import within_calling_hours
from agent.history import (
    failed_verifications,
    record_call,
    record_verification_attempt,
    summarise_for_agent,
)
from agent.redact import MASK, redact, redact_text
from agent.tools import run_tool


@pytest.fixture
def db(tmp_path):
    return tmp_path / "history.db"


def new_call(db, bid="LF-1002"):
    return CallState(borrower=get_borrower(bid), history_db=db)


# --- promise_to_pay_date ------------------------------------------------------------------


END = {"goodbye": "Bye.", "outcome": "promise_to_pay", "escalate_to_human": False, "notes": "Paying soon."}


@pytest.mark.parametrize("bad", ["next Friday", "2026-02-30", (date.today() - timedelta(days=1)).isoformat()])
async def test_invalid_promise_date_is_refused_before_anything_is_recorded(db, bad):
    state = new_call(db)
    state.verified = True
    result = await run_tool(state, "end_call", {**END, "promise_to_pay_date": bad})
    assert "error" in result and "Confirm the exact date" in result["error"]
    assert not state.ended  # the bot hands the error back to the LLM, which asks and calls again
    good = (date.today() + timedelta(days=7)).isoformat()
    await run_tool(state, "end_call", {**END, "promise_to_pay_date": good})
    assert state.outcome.promise_to_pay_date == good and state.outcome.notes == "Paying soon."


async def test_unknown_outcome_is_refused_so_the_call_is_not_hung_up(db):
    state = new_call(db)
    result = await run_tool(state, "end_call", {"goodbye": "Bye.", "outcome": "paid_maybe",
                                                "escalate_to_human": False, "notes": ""})
    assert "error" in result and not state.ended  # the bot hands this back to the LLM


def test_summary_skips_a_bad_stored_date_instead_of_crashing():
    calls = [{"created_at": "2026-11-03T11:20:00", "outcome": "promise_to_pay", "promise_to_pay_date": "next Friday"}]
    [item] = summarise_for_agent(calls, today=date(2026, 11, 16))
    assert item["outcome"] == "promise_to_pay" and "promised_to_pay_on" not in item


async def test_bad_date_from_an_old_call_does_not_block_verification(db):
    record_call({"call_id": "old", "borrower_id": "LF-1002", "outcome": "promise_to_pay", "verified": 1,
                 "promise_to_pay_date": "pixel", "created_at": "2026-11-03T11:20:00"}, db)
    state = new_call(db)
    result = await run_tool(state, "verify_identity", {"phone_last4": "6789"})
    assert result["verified"] is True and result["previous_calls"][0]["outcome"] == "promise_to_pay"


# --- verified only after success ----------------------------------------------------------


async def test_verified_is_set_only_after_the_result_is_built(db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("formatter broke")

    monkeypatch.setattr("agent.tools.rupees_to_words", boom)
    state = new_call(db)
    result = await run_tool(state, "verify_identity", {"phone_last4": "6789"})
    assert "error" in result and state.verified is False
    assert "error" in await run_tool(state, "send_payment_link", {})  # gate stays shut


async def test_history_outage_still_verifies(db, monkeypatch):
    def down(*a, **k):
        raise OSError("disk unavailable")

    monkeypatch.setattr("agent.tools.recent_calls", down)
    state = new_call(db)
    result = await run_tool(state, "verify_identity", {"phone_last4": "6789"})
    assert result["verified"] is True and "loan" in result and "previous_calls" not in result


# --- verification attempts ----------------------------------------------------------------


async def test_one_factor_per_attempt_and_empty_checks_do_not_count(db):
    state = new_call(db)
    both = await run_tool(state, "verify_identity", {"dob": "2000-01-01", "phone_last4": "6789"})
    neither = await run_tool(state, "verify_identity", {"dob": "", "phone_last4": None})
    assert both["verified"] is False and neither["verified"] is False
    assert state.verification_attempts == 0
    assert (await run_tool(state, "verify_identity", {"phone_last4": "6789"}))["verified"] is True


async def test_attempts_remaining_counts_down_from_the_limit(db):
    state = new_call(db)
    first = await run_tool(state, "verify_identity", {"dob": "2000-01-01"})
    second = await run_tool(state, "verify_identity", {"dob": "2000-01-02"})
    assert (first["attempts_remaining"], second["attempts_remaining"]) == (1, 0)


async def test_attempts_remaining_counts_the_cross_call_budget(db):
    call1 = new_call(db)
    await run_tool(call1, "verify_identity", {"dob": "2000-01-01"})
    await run_tool(call1, "verify_identity", {"dob": "2000-01-02"})
    call2 = new_call(db)  # 1 try left this call by itself, but the 3rd failure overall locks out
    first = await run_tool(call2, "verify_identity", {"dob": "2000-01-03"})
    assert first["attempts_remaining"] == 0


async def test_failures_lock_out_across_calls(db):
    call1 = new_call(db)
    await run_tool(call1, "verify_identity", {"dob": "2000-01-01"})
    await run_tool(call1, "verify_identity", {"dob": "2000-01-02"})
    call2 = new_call(db)  # a fresh call gets its own two tries, but the borrower has a budget too
    await run_tool(call2, "verify_identity", {"dob": "2000-01-03"})
    right = await run_tool(call2, "verify_identity", {"phone_last4": "6789"})  # correct, but locked
    assert right["verified"] is False and "loan" not in right and call2.verification_locked
    call3 = new_call(db)
    assert (await run_tool(call3, "verify_identity", {"phone_last4": "6789"}))["verified"] is False


async def test_a_success_resets_the_cross_call_count(db):
    call1 = new_call(db)
    await run_tool(call1, "verify_identity", {"dob": "2000-01-01"})
    await run_tool(call1, "verify_identity", {"phone_last4": "6789"})
    assert failed_verifications("LF-1002", 24, db) == 0


def test_old_failures_age_out(db):
    old = datetime.now(UTC) - timedelta(hours=30)
    for _ in range(5):
        record_verification_attempt("LF-1002", "c-old", False, db, now=old)
    assert failed_verifications("LF-1002", 24, db) == 0


# --- PII in logs --------------------------------------------------------------------------


def test_redact_masks_tool_args_and_free_text_but_keeps_call_ids():
    args = redact({"dob": "1990-03-14", "phone_last4": "2345", "outcome": "promise_to_pay"})
    assert args == {"dob": MASK, "phone_last4": MASK, "outcome": "promise_to_pay"}
    line = redact_text("User said: my birthday is 14/03/1990, mobile ends two three four five, or 2345. "
                       "Call 20261121T101010123456, 1.09s")
    said, _, rest = line.partition("Call ")
    assert "1990" not in said and "2345" not in said and "two three four five" not in said
    assert rest == "20261121T101010123456, 1.09s"


async def test_real_verify_result_logs_no_loan_details(db):
    import json

    state = new_call(db, "LF-1001")
    result = await run_tool(state, "verify_identity", {"dob": "1990-03-14"})
    assert result["verified"] is True and result["loan"]["emi_amount_inr"] == 12500
    logged = json.dumps(redact(result), ensure_ascii=False)
    assert redact(result)["loan"] == MASK
    for leak in ("12500", "twelve thousand", "बारह", "fifth of December", "Personal Loan"):
        assert leak not in logged


async def test_payment_lookup_logs_no_amounts(db):
    import json

    state = new_call(db, "LF-1008")
    await run_tool(state, "verify_identity", {"phone_last4": "0011"})
    logged = json.dumps(redact(await run_tool(state, "check_payment_status", {})), ensure_ascii=False)
    assert "7200" not in logged and "seven thousand" not in logged


@pytest.mark.parametrize("line", [
    # Pipecat's DEBUG lines carry the LLM context and every TTS line in plain text.
    "Generating chat: [{'role': 'assistant', 'content': 'Your EMI of twelve thousand five hundred rupees is due on the fifth of December.'}]",
    "TTS: overdue 4,850 (four thousand eight hundred and fifty) since 10 November",
    "TTS: आपके पर्सनल लोन की बारह हज़ार पाँच सौ रुपये की किस्त पाँच दिसंबर को देय है",
    "TTS: बारह हजार पांच सौ रुपए",  # written without nukta or chandrabindu
])
def test_spoken_amounts_and_due_dates_are_masked_in_log_lines(line):
    out = redact_text(line)
    for leak in ("twelve thousand", "fifth of December", "4,850", "four thousand", "10 November", "बारह", "दिसंबर"):
        assert leak not in out


def test_redact_reaches_nested_results():
    out = redact({"verified": False, "history": [{"notes": "Call back on 98765 43210"}]})
    assert "98765" not in out["history"][0]["notes"]
    assert redact({"previous_calls": [{"notes": "x"}]})["previous_calls"] == MASK  # loan info: masked whole


# --- payment link -------------------------------------------------------------------------


async def test_payment_link_is_sent_once_per_call(db, monkeypatch):
    monkeypatch.setattr("agent.tools.CONFIG", type("C", (), {"payment_link_delay_secs": 0.01,
                                                             "payment_link_failure_rate": 0})())
    state = new_call(db)
    state.verified = True
    a, b = await asyncio.gather(run_tool(state, "send_payment_link", {}), run_tool(state, "send_payment_link", {}))
    c = await run_tool(state, "send_payment_link", {})
    assert a["link_id"] == b["link_id"] == c["link_id"]
    assert sum(1 for r in (a, b, c) if not r.get("already_sent")) == 1


# --- calling hours ------------------------------------------------------------------------


def test_outside_hours_line_in_each_language_reveals_nothing():
    from agent.templates import outside_hours
    from evals.metrics import leak_patterns

    for bid, word in (("LF-1001", "calling hours"), ("LF-1003", "कॉल करने का समय")):
        b = get_borrower(bid)
        line = outside_hours(b)
        assert word in line and "Lumen Finance" in line
        assert not any(p in line.lower() for p in leak_patterns(b))


@pytest.mark.parametrize("ist, allowed", [("07:59", False), ("08:00", True), ("13:30", True),
                                          ("18:59", True), ("19:00", False), ("23:00", False)])
def test_calling_hours_in_ist(ist, allowed):
    h, m = map(int, ist.split(":"))
    utc = datetime(2026, 11, 21, h, m, tzinfo=UTC) - timedelta(hours=5, minutes=30)
    assert within_calling_hours(utc, "08:00-19:00") is allowed


# --- admin page ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import agent.web as web

    monkeypatch.setattr(web, "ADMIN_TOKEN", "test-token")
    return TestClient(web.app), web


def test_admin_endpoints_need_the_token(client):
    c, _ = client
    for method, path in [("get", "/borrowers"), ("get", "/api/borrowers"),
                         ("post", "/api/borrower/LF-1001/clear-history")]:
        assert getattr(c, method)(path).status_code == 401
        assert getattr(c, method)(path, headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert c.get("/api/borrowers", headers={"X-Admin-Token": "test-token"}).status_code == 200
    assert c.get("/borrowers?token=test-token").status_code == 200


def test_clear_history_rejects_unknown_borrowers(client):
    c, _ = client
    assert c.post("/api/borrower/nope/clear-history", headers={"X-Admin-Token": "test-token"}).status_code == 404


def test_page_data_cannot_break_out_of_the_script(client, monkeypatch):
    c, web = client
    evil = "</script><img src=x onerror=alert(1)>"
    monkeypatch.setattr(web, "_borrower_rows", lambda: [{"id": "LF-1", "name": evil, "loan": "", "language": "",
                                                          "dob": "", "last4": "", "amount": "", "status": "",
                                                          "history": []}])
    body = c.get("/borrowers?token=test-token").text
    assert evil not in body and "\\u003c/script\\u003e" in body
    assert "esc(x.name)" in body  # rendered through the escaper, not raw innerHTML
