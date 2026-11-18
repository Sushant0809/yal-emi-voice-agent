from datetime import date

import pytest

from agent.borrowers import check_identity, get_borrower
from agent.call_state import CallState, Outcome
from agent.history import recent_calls, record_call, summarise_for_agent
from agent.speech import number_to_words, to_spoken
from agent.tools import run_tool


@pytest.fixture
def db(tmp_path):
    return tmp_path / "history.db"


@pytest.mark.parametrize(
    "n, words",
    [
        (12500, "twelve thousand five hundred"),
        (4850, "four thousand eight hundred fifty"),
        (125000, "one lakh twenty-five thousand"),
        (38200, "thirty-eight thousand two hundred"),
        (10_000_000, "one crore"),
    ],
)
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_to_spoken():
    assert to_spoken("₹12,500 due on 2026-12-05.") == "twelve thousand five hundred rupees due on the fifth of December."
    assert to_spoken("Rs. 2,999/-") == "two thousand nine hundred ninety-nine rupees"
    assert to_spoken("आपकी EMI ₹38,200 है") == "आपकी EMI अड़तीस हज़ार दो सौ रुपये है"  # verbalised in Hindi


def test_check_identity():
    b = get_borrower("LF-1001")
    assert check_identity(b, dob="1990-03-14")
    assert check_identity(b, dob="14/03/1990")
    assert check_identity(b, phone_last4="2345")
    assert not check_identity(b, dob="1990-01-01")
    assert not check_identity(b)


async def test_loan_details_only_after_verification(db):
    state = CallState(borrower=get_borrower("LF-1002"), history_db=db)
    bad = await run_tool(state, "verify_identity", {"dob": "2000-01-01"})
    assert bad["verified"] is False and "loan" not in bad
    link = await run_tool(state, "send_payment_link", {})
    assert "error" in link and not state.payment_link_sent  # refused by the generic gate

    good = await run_tool(state, "verify_identity", {"phone_last4": "6789"})
    assert good["verified"] is True
    assert good["loan"]["overdue_amount_spoken"] == "four thousand eight hundred fifty rupees"


async def test_verification_locks_after_two_failures(db):
    state = CallState(borrower=get_borrower("LF-1002"), history_db=db)
    await run_tool(state, "verify_identity", {"dob": "2000-01-01"})
    await run_tool(state, "verify_identity", {"dob": "2000-01-02"})
    third = await run_tool(state, "verify_identity", {"phone_last4": "6789"})
    assert third["verified"] is False and state.verification_locked


async def test_hardship_always_escalates(db):
    state = CallState(borrower=get_borrower("LF-1004"), history_db=db)
    await run_tool(state, "end_call", {"outcome": "hardship", "escalate_to_human": False, "notes": "Lost job."})
    assert state.outcome.escalate_to_human is True
    # Second end_call does not overwrite the first record
    await run_tool(state, "end_call", {"outcome": "will_pay_on_time", "escalate_to_human": False, "notes": ""})
    assert state.outcome.outcome is Outcome.HARDSHIP


def _promise(call_id, created_at, ptp="2026-11-10"):
    return {
        "call_id": call_id, "borrower_id": "LF-1006", "outcome": "promise_to_pay", "verified": 1,
        "promise_to_pay_date": ptp, "notes": "Salary delayed.", "created_at": created_at,
    }


def test_history_is_idempotent_and_newest_first(db):
    record_call(_promise("c1", "2026-11-01T10:00:00"), db)
    record_call(_promise("c2", "2026-11-05T10:00:00", ptp="2026-11-12"), db)
    record_call({**_promise("c1", "2026-11-01T10:00:00"), "outcome": "dispute"}, db)  # retry: ignored
    calls = recent_calls("LF-1006", db_path=db)
    assert [c["call_id"] for c in calls] == ["c2", "c1"]
    assert calls[1]["outcome"] == "promise_to_pay"
    assert recent_calls("LF-1001", db_path=db) == []


def test_summary_flags_broken_promise():
    calls = [_promise("c1", "2026-11-03T11:20:00")]
    [item] = summarise_for_agent(calls, today=date(2026, 11, 16))
    assert item["when"] == "13 days ago (the third of November)"
    assert item["promised_to_pay_on"] == "the tenth of November"
    assert item["promise_date_has_passed"] is True


async def test_history_only_released_after_verification(db):
    record_call(_promise("c1", "2026-11-03T11:20:00"), db)
    state = CallState(borrower=get_borrower("LF-1006"), history_db=db)
    failed = await run_tool(state, "verify_identity", {"dob": "2000-01-01"})
    assert "previous_calls" not in failed
    ok = await run_tool(state, "verify_identity", {"dob": "1992-12-17"})
    assert ok["previous_calls"][0]["promised_to_pay_on"] == "the tenth of November"


async def test_end_call_persists_to_history(db):
    state = CallState(borrower=get_borrower("LF-1001"), history_db=db)
    await run_tool(state, "end_call", {"outcome": "will_pay_on_time", "escalate_to_human": False, "notes": "Paying on the 5th."})
    state.save()
    state.save()  # second save is a no-op
    [row] = recent_calls("LF-1001", db_path=db)
    assert row["outcome"] == "will_pay_on_time" and row["call_id"] == state.call_id


def test_closing_line_speaks_goodbye_field():
    from agent.tools import closing_line
    args = {"goodbye": "I've sent the link. Thank you, Rahul. Goodbye."}
    # Model's text was hidden (Sonnet 5.5 returned it as a thinking block): speak the field.
    assert closing_line(args, "", "en") == args["goodbye"]
    # Model already said goodbye as text: don't say it twice.
    assert closing_line(args, "Thanks Rahul, goodbye!", "en") is None
    # Model spoke its own closing without "bye": still don't overlap it (garbled live call).
    assert closing_line(args, "Once you've paid through the link, you're all set.", "en") is None
    # Field missing (should not happen, it's required): never hang up in silence.
    assert closing_line({}, "", "hi") == "आपका समय देने के लिए धन्यवाद। नमस्ते।"
