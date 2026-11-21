from datetime import date, timedelta

from agent.borrowers import get_borrower
from agent.templates import greeting, reminder


def test_greeting_by_language():
    assert greeting(get_borrower("LF-1001"), recorded=False) == (
        "Hello, this is Asha calling from Lumen Finance. Am I speaking with Rahul Sharma?")


def test_recorded_calls_say_so_before_asking_anything():
    en = greeting(get_borrower("LF-1001"), recorded=True)
    assert en == ("Hello, this is Asha calling from Lumen Finance. This call is recorded for quality and "
                  "training. Am I speaking with Rahul Sharma?")
    hi = greeting(get_borrower("LF-1003"), recorded=True)
    assert "रिकॉर्ड" in hi and hi.index("रिकॉर्ड") < hi.index("क्या मेरी बात")
    assert "Lumen Finance" in greeting(get_borrower("LF-1003")) and "अमित" not in greeting(get_borrower("LF-1003"))


def test_reminder_due_and_overdue():
    ok = {"verified": True}
    assert reminder(get_borrower("LF-1001"), ok, "en") == (
        "Thanks, Rahul, you're verified. Your personal loan EMI of twelve thousand five hundred rupees "
        "is due on the fifth of December. Will you be able to pay it by then?"
    )
    over = reminder(get_borrower("LF-1002"), ok, "en")
    assert "four thousand eight hundred fifty rupees was due on the tenth of November" in over
    assert "eight days overdue" in over


def test_due_dates_move_with_today_so_the_data_never_goes_stale(monkeypatch):
    from agent import dates

    priya, rahul = get_borrower("LF-1002"), get_borrower("LF-1001")  # 8 days overdue; due in 17 days
    for today in (date(2026, 11, 18), date(2027, 3, 1)):
        monkeypatch.setattr(dates, "today", lambda today=today: today)
        assert date.fromisoformat(priya.due_date) == today - timedelta(days=8)
        assert priya.days_overdue == 8 and "eight days overdue" in reminder(priya, {"verified": True}, "en")
        assert priya.loan_details()["due_date"] == priya.due_date  # what the LLM sees agrees
        assert date.fromisoformat(rahul.due_date) - today == timedelta(days=17) and not rahul.is_overdue


def test_upcoming_emi_with_nothing_unpaid_is_not_overdue():
    ananya = get_borrower("LF-1008")  # due in 2 days, already paid (processing)
    assert not ananya.is_overdue and ananya.days_overdue == 0


def test_llm_writes_it_when_judgement_is_needed():
    b = get_borrower("LF-1006")
    assert reminder(b, {"verified": False}, "en") is None  # failed verification
    assert reminder(b, {"verified": True, "previous_calls": [{"outcome": "promise_to_pay"}]}, "en") is None  # history

def test_hindi_reminder_is_templated_with_hindi_numbers():
    line = reminder(get_borrower("LF-1003"), {"verified": True}, "hi")
    assert "अड़तीस हज़ार दो सौ रुपये" in line and "एक दिसंबर" in line and "होम लोन" in line
