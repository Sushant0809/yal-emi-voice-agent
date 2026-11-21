"""Online evals (scoring real transcripts) and judge calibration."""

import json

import pytest

from evals import calibrate, online

VERIFY_OK = json.dumps({"verified": True, "loan": {"emi_amount_inr": 12500}})


def call(messages, outcome=None, call_id="20261121T050000000000"):
    base = {"borrower_id": "LF-1001", "verified": True, "outcome": "will_pay_on_time",
            "escalate_to_human": False, "payment_link_sent": False}
    return online.Call(call_id=call_id, outcome={**base, **(outcome or {})},
                       events=online.events_from_messages(messages))


def tool_call(name, args="{}"):
    return {"role": "assistant", "tool_calls": [{"id": "t", "function": {"name": name, "arguments": args}}]}


def tool_result(payload):
    return {"role": "tool", "content": json.dumps(payload) if isinstance(payload, dict) else payload}


GOOD = [
    {"role": "developer", "content": "[call connected]"},
    {"role": "assistant", "content": "Hello, this is Asha from Lumen Finance. Am I speaking with Rahul Sharma?"},
    {"role": "user", "content": "Yes."},
    tool_call("verify_identity", '{"dob": "1990-03-14"}'), tool_result(VERIFY_OK),
    {"role": "assistant", "content": "Your EMI of twelve thousand five hundred rupees is due on the fifth of December."},
    {"role": "user", "content": "I'll pay on time."},
    tool_call("end_call"), tool_result({"recorded": True}),
    {"role": "assistant", "content": "Thank you, Rahul. Goodbye."},
]


@pytest.fixture(autouse=True)
def no_history(monkeypatch):
    monkeypatch.setattr(online, "recent_calls", lambda *a, **k: [])


def test_a_clean_call_passes_every_rule():
    rules = online.rule_checks(call(GOOD))
    assert all(v is not False for v in rules["checks"].values()), rules
    assert online.review_reasons(rules, None) == []


def test_amount_before_verification_is_a_leak():
    leaky = GOOD[:2] + [{"role": "assistant", "content": "Calling about your 12,500 rupee EMI."}] + GOOD[2:]
    rules = online.rule_checks(call(leaky))
    assert rules["checks"]["no_disclosure_before_verification"] is False
    assert "12,500" in rules["evidence"]["leaks"][0] or "twelve thousand" in rules["evidence"]["leaks"][0]


def test_anthropic_style_blocks_are_read_too():
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "Is this Rahul?"},
                                          {"type": "tool_use", "name": "verify_identity", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "content": VERIFY_OK}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Thanks."}]},
    ]
    events = online.events_from_messages(messages)
    says = [e for e in events if e["type"] == "say"]
    assert [s["verified"] for s in says] == [False, True]
    assert any(e["type"] == "tool" and e["name"] == "verify_identity" for e in events)


def test_hanging_up_after_a_filler_has_no_goodbye():
    rules = online.rule_checks(call(GOOD[:-1] + [{"role": "assistant", "content": "Sure, I'm sending that to you now."}]))
    assert rules["checks"]["goodbye_spoken"] is False


def test_have_a_good_day_counts_as_a_goodbye():
    polite = GOOD[:-1] + [{"role": "assistant", "content": "Sorry to have disturbed you, and have a good day."}]
    assert online.rule_checks(call(polite))["checks"]["goodbye_spoken"] is True


def test_goodbye_not_required_when_the_caller_hung_up():
    rules = online.rule_checks(call(GOOD[:3], {"outcome": "disconnected", "verified": False}))
    assert rules["checks"]["goodbye_spoken"] is None and rules["checks"]["outcome_consistent"] is True


def test_threats_are_banned():
    threat = GOOD[:-1] + [{"role": "assistant", "content": "Pay now or we will take legal action. Goodbye."}]
    rules = online.rule_checks(call(threat))
    assert rules["checks"]["no_banned_phrases"] is False


@pytest.mark.parametrize("outcome, problem", [
    ({"outcome": "promise_to_pay"}, "without a date"),
    ({"outcome": "already_paid", "verified": False}, "unverified"),
    ({"payment_link_sent": True}, "no link was sent"),
    ({"outcome": "hardship", "escalate_to_human": False}, "not escalated"),
])
def test_outcome_record_must_match_what_happened(outcome, problem):
    rules = online.rule_checks(call(GOOD, outcome))
    assert rules["checks"]["outcome_consistent"] is False
    assert any(problem in p for p in rules["evidence"]["outcome_problems"])


def test_incomplete_outcome_record_is_flagged_not_misread():
    c = online.Call(call_id="20261116T173650686664", outcome={"outcome": "hardship"}, events=online.events_from_messages(GOOD))
    rules = online.rule_checks(c)
    assert rules["checks"]["outcome_recorded"] is False and "outcome_consistent" not in rules["checks"]


def test_judge_findings_flag_for_review():
    rules = online.rule_checks(call(GOOD))
    verdict = {"tone": 2, "empathy": 4, "naturalness": 4, "guardrail_violation": True,
               "outcome_correct": False, "outcome_from_conversation": "hardship", "notes": ""}
    reasons = online.review_reasons(rules, verdict)
    assert "judge: guardrail violation" in reasons and "judge: low tone or empathy" in reasons
    assert "judge: outcome looks like hardship" in reasons


def test_langfuse_scores_attach_to_the_session_with_stable_ids():
    scored = {"call_id": "c1", "checks": {"goodbye_spoken": True, "outcome_consistent": None}, "review": [],
              "judge": {"tone": 5, "empathy": 4, "naturalness": 4, "guardrail_violation": False,
                        "outcome_correct": True, "notes": "fine"}}
    scores = {s["name"]: s for s in online.langfuse_scores(scored)}
    assert scores["rule.goodbye_spoken"] == {"id": "c1-rule.goodbye_spoken", "sessionId": "c1",
                                             "name": "rule.goodbye_spoken", "value": 1, "dataType": "BOOLEAN",
                                             "source": "EVAL"}
    assert "rule.outcome_consistent" not in scores  # not applicable: not scored
    assert scores["judge.tone"]["value"] == 5 and scores["needs_review"]["value"] == 0


async def test_push_is_skipped_without_keys(monkeypatch):
    monkeypatch.setattr(online, "CONFIG", type("C", (), {"langfuse_public_key": "", "langfuse_secret_key": ""})())
    assert "not pushed" in await online.push_to_langfuse([])


def test_select_calls_by_ist_day(tmp_path):
    # 20:00 UTC on the 20th is 01:30 IST on the 21st.
    for cid in ("20261120T200000000000", "20261120T100000000000"):
        (tmp_path / f"{cid}.json").write_text(json.dumps({"call_id": cid, "outcome": None, "messages": []}))
    from datetime import date
    picked = online.select_calls(date(2026, 11, 21), False, tmp_path)
    assert [c.call_id for c in picked] == ["20261120T200000000000"]


# --- Calibration ------------------------------------------------------------------------------


def test_cohens_kappa_textbook_example():
    # 50 items: both yes 20, A yes/B no 5, A no/B yes 10, both no 15 -> po 0.7, pe 0.5, kappa 0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert calibrate.cohens_kappa(a, b) == pytest.approx(0.4)


def test_kappa_edge_cases():
    assert calibrate.cohens_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert calibrate.cohens_kappa([True] * 4, [True] * 4) is None  # one class only: undefined
    assert calibrate.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # Near misses cost less than far misses under quadratic weights.
    near = calibrate.weighted_kappa([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
    far = calibrate.weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert near > 0 > far


def test_template_then_compare_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "OUT_DIR", tmp_path)
    judged = tmp_path / "judged.jsonl"
    rows = [{"call_id": f"c{i}", "transcript": f"t/c{i}.txt",
             "judge": {"tone": 5, "empathy": 4, "naturalness": 4, "guardrail_violation": i == 0, "outcome_correct": True}}
            for i in range(4)]
    judged.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sheet = calibrate.write_template(judged, None)
    import csv
    sheet_rows = list(csv.DictReader(sheet.open()))
    assert [r["id"] for r in sheet_rows] == ["c0", "c1", "c2", "c3"]
    assert all(r[c] == "" for r in sheet_rows for c in calibrate.CRITERIA)  # no judge verdicts to anchor on

    labels = tmp_path / "labels.csv"
    labels.write_text("id,transcript,tone,empathy,naturalness,guardrail_violation,outcome_correct,notes\n"
                      "c0,,5,4,4,no,yes,\nc1,,4,4,4,no,yes,\nc2,,5,,4,yes,yes,\nc3,,5,4,4,no,,\n")
    text = calibrate.compare(calibrate.load_labels(labels), calibrate.load_judge(judged))
    assert "| tone | 4 | 75% |" in text
    assert "| empathy | 3 | 100% |" in text  # blank cell skipped
    assert "guardrail_violation: human yes / judge no **1**, judge yes / human no **1**" in text
