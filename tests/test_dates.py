"""The mock world is relative to today, so data and eval scenarios never go stale."""

import json
from datetime import date, timedelta

import pytest

from agent import dates
from agent.config import ROOT


def test_date_tokens():
    anchor = date(2026, 11, 18)
    assert dates.resolve("{+7}", anchor) == "2026-11-25"
    assert dates.resolve("{-15}T11:20:00", anchor) == "2026-11-03T11:20:00"
    assert dates.resolve("pay on {+7:th}, the {+7:day_th}", anchor) == "pay on 25th November, the 25th"
    assert dates.resolve("{-8:spoken}|the {-8:ordinal}", anchor) == "the tenth of November|the tenth"
    assert dates.resolve("{-8:hi}", anchor) == "दस नवंबर"
    assert dates.resolve({"a": ["{+1}", 3]}, anchor) == {"a": ["2026-11-19", 3]}
    assert dates.resolve("{lender} stays {as is}", anchor) == "{lender} stays {as is}"  # only {±N} tokens
    assert [dates.resolve(f"{{+{n}:day_th}}", date(2026, 11, 30)) for n in (1, 2, 3, 11, 21)] == \
        ["1st", "2nd", "3rd", "11th", "21st"]


@pytest.mark.parametrize("today", [date(2026, 11, 21), date(2026, 12, 28), date(2027, 2, 25)])  # month/year ends
def test_scenarios_stay_valid_on_any_day(monkeypatch, today):
    from evals.run_evals import load_scenarios

    monkeypatch.setattr(dates, "today", lambda: today)
    for s in load_scenarios():
        assert not dates.TOKEN.search(json.dumps(s, ensure_ascii=False)), s["id"]  # every token filled
        if ptp := s.get("expected", {}).get("promise_to_pay_date"):
            assert date.fromisoformat(ptp) > today, s["id"]  # end_call would refuse a past date
        for h in s.get("history", []):
            if h.get("promise_to_pay_date"):
                assert date.fromisoformat(h["promise_to_pay_date"]) < today, s["id"]  # a broken promise


def test_seed_history_is_relative(monkeypatch, tmp_path):
    from agent.history import recent_calls, seed

    monkeypatch.setattr(dates, "today", lambda: date(2027, 1, 10))
    seed(ROOT / "data" / "call_history_seed.json", tmp_path / "h.db")
    [kavya] = recent_calls("LF-1006", db_path=tmp_path / "h.db")
    assert kavya["promise_to_pay_date"] == "2027-01-02" and kavya["created_at"].startswith("2026-12-26")
    assert kavya["notes"].endswith("promised to pay on the 2nd. Payment link sent.")


async def test_payment_dates_are_relative(monkeypatch, tmp_path):
    from agent.borrowers import get_borrower
    from agent.call_state import CallState
    from agent.tools import run_tool

    monkeypatch.setattr(dates, "today", lambda: date(2027, 1, 10))
    state = CallState(borrower=get_borrower("LF-1008"), history_db=tmp_path / "h.db")
    await run_tool(state, "verify_identity", {"phone_last4": "0011"})
    [latest, _] = (await run_tool(state, "check_payment_status", {}))["recent_payments"]
    assert latest["date"] == (date(2027, 1, 10) - timedelta(days=1)).isoformat() and latest["status"] == "processing"
