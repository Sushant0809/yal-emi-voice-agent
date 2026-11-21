import pytest


@pytest.fixture(autouse=True)
def _audit_to_tmp(tmp_path, monkeypatch):
    """Tests must never write to the real audit log in outcomes/."""
    from agent import audit

    monkeypatch.setattr(audit, "AUDIT_PATH", tmp_path / "audit.jsonl")


@pytest.fixture(autouse=True)
def _pinned_today(monkeypatch):
    """Mock-world dates are relative to today (agent/dates.py). Pin today to 18 Nov 2026, the day
    the data was written, so expected strings ("due on the fifth of December") stay exact."""
    from datetime import date

    from agent import dates

    monkeypatch.setattr(dates, "today", lambda: date(2026, 11, 18))
