"""Latency percentiles, interruption rate, STT accuracy scoring and the CI eval gate."""

import pytest

from evals import latency_report as lr
from evals import wer


def test_nearest_rank_percentiles():
    v = [float(x) for x in range(1, 21)]  # 1..20
    assert (lr._pct(v, 0.5), lr._pct(v, 0.9), lr._pct(v, 0.95)) == (10.0, 18.0, 19.0)
    assert lr._pct([3.0], 0.95) == 3.0


def test_interruption_rate_ignores_the_turn_open_at_hang_up():
    log = [
        {"call_id": "a", "was_interrupted": True, "call_ended": False},
        {"call_id": "a", "was_interrupted": False, "call_ended": False},
        {"call_id": "a", "was_interrupted": True, "call_ended": True},   # hang-up, not a barge-in
        {"call_id": "b", "was_interrupted": False, "call_ended": False},
    ]
    assert lr.interruptions(log) == ("Interruption rate: 1/3 turns (33%) were cut off by the caller; "
                                     "1 of 2 calls had at least one.")
    assert "no turn data" in lr.interruptions([])


@pytest.mark.parametrize("entity, heard, ok", [
    ({"digits": "2345"}, "The last four digits are 2345.", True),
    ({"digits": "2345"}, "two three four five", True),
    ({"digits": "2345"}, "दो तीन चार पांच", True),
    ({"digits": "2345"}, "two three four", False),
    ({"amount": 12500}, "I'll pay 12,500 rupees", True),
    ({"amount": 12500}, "बारह हजार पांच सौ रुपए", True),
    ({"amount": 4850}, "four thousand eight hundred and fifty", True),
    ({"amount": 4850}, "four thousand five hundred", False),
    ({"date": "1990-03-14"}, "fourteenth March nineteen ninety", True),
    ({"date": "1990-03-14"}, "मेरी जन्मतिथि fourteen March उन्नीस सौ नब्बे है", True),
    ({"date": "1990-03-14"}, "fourteenth March nineteen eighty", False),
    ({"date": "--12-05"}, "on the 5th of December", True),
    ({"date": "--12-05"}, "on the fifth of November", False),
    ({"day": 27}, "by the twenty seventh", True),
])
def test_entities_match_in_any_spelling(entity, heard, ok):
    assert wer.entity_found(entity, heard) is ok


def test_wer_raw_and_normalised():
    ref = "मैं पाँच दिसंबर तक भर दूँगा"
    hyp = "मैं पांच December तक भर दूंगा."
    assert wer.wer(ref, hyp) == pytest.approx(1 / 6)      # "December" for "दिसंबर" counts as an error
    assert wer.wer(ref, hyp, canonical=True) == 0          # ...but not once both mean month 12
    assert wer.wer("two three four five", "two three five") == pytest.approx(0.25)


def test_every_reference_case_scores_perfectly_against_itself():
    import yaml

    for case in yaml.safe_load(wer.CASES_PATH.read_text())["cases"]:
        r = wer.score(case, case["text"])
        assert r["wer"] == 0 and all(e["found"] for e in r["entities"]), case["id"]


def test_eval_summary_names_the_prompt_version_and_pass_rate():
    from agent.prompts import prompt_version
    from evals.run_evals import summarise

    runs = [{"scenario": "s", "run": i, "checks": {"ok": i < 3}, "turns": 4, "llm_secs": []} for i in range(4)]
    assert summarise(runs).startswith(f"Prompt version {prompt_version()}: 3/4 runs pass (75%).")


def test_ci_subset_is_tagged():
    import yaml

    from evals.run_evals import SCENARIOS_PATH

    ci = [s["id"] for s in yaml.safe_load(SCENARIOS_PATH.read_text())["scenarios"] if "ci" in s.get("subsets", [])]
    assert {"failed_verification", "family_member", "prompt_injection"} <= set(ci)
