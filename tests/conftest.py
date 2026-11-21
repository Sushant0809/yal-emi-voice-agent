import pytest


@pytest.fixture(autouse=True)
def _audit_to_tmp(tmp_path, monkeypatch):
    """Tests must never write to the real audit log in outcomes/."""
    from agent import audit

    monkeypatch.setattr(audit, "AUDIT_PATH", tmp_path / "audit.jsonl")
