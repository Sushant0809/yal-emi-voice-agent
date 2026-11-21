"""Run every scenario N times: text agent vs LLM borrower, then score.

    uv run python -m evals.run_evals                 # all scenarios, 3 runs each
    uv run python -m evals.run_evals -s hardship -n 5
    uv run python -m evals.run_evals --no-judge      # rule checks only (cheaper)
    uv run python -m evals.run_evals --subset ci --no-judge -n 1 --min-pass-rate 0.9   # the CI gate

Writes results/<timestamp>/{runs.jsonl, summary.md}.
"""

import argparse
import asyncio
import json
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import anthropic
import yaml

from agent.borrowers import get_borrower
from agent.call_state import Outcome
from agent.config import ROOT
from agent.history import record_call
from agent.prompts import prompt_version
from evals.metrics import judge, rule_checks
from evals.simulator import HANGUP, BorrowerSimulator
from evals.text_agent import PROVIDERS, OpenAICompatTextAgent, TextAgent

MAX_TURNS = 14
SCENARIOS_PATH = ROOT / "evals" / "scenarios.yaml"


def seed_history(scenario: dict, db_path: Path) -> None:
    """Load the scenario's prior calls (if any) into this run's own database."""
    for i, call in enumerate(scenario.get("history", [])):
        created = datetime.now() - timedelta(days=call.get("days_ago", 7))
        record_call(
            {
                "call_id": f"hist-{scenario['id']}-{i}",
                "borrower_id": scenario["borrower_id"],
                "verified": 1,
                "created_at": created.isoformat(timespec="seconds"),
                **{k: v for k, v in call.items() if k != "days_ago"},
            },
            db_path,
        )


async def run_one(client, scenario: dict, run_idx: int, use_judge: bool, model: str | None = None,
                  thinking: str | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "history.db"
        seed_history(scenario, db_path)
        return await _run_call(client, scenario, run_idx, use_judge, db_path, model, thinking)


async def _run_call(client, scenario: dict, run_idx: int, use_judge: bool, db_path: Path,
                    model: str | None, thinking: str | None) -> dict:
    borrower = get_borrower(scenario["borrower_id"])
    if model and model.partition(":")[0] in PROVIDERS:
        agent = OpenAICompatTextAgent(borrower, db_path, model)
    else:
        agent = TextAgent(client, borrower, db_path, model, thinking)
    agent.state.inject_failures = dict(scenario.get("inject_failures", {}))
    sim = BorrowerSimulator(client, scenario, scenario.get("facts", ""))
    transcript: list[tuple[str, str]] = []

    agent_text = await agent.start()
    transcript.append(("AGENT", agent_text))
    for _ in range(MAX_TURNS):
        if agent.state.ended:
            break
        caller = await sim.reply(agent_text)
        if caller.strip() == HANGUP:
            transcript.append(("BORROWER", "(hangs up)"))
            if not agent.state.ended:  # same as the live bot's on_client_disconnected
                agent.state.finalize(Outcome.DISCONNECTED, notes="Caller hung up before the call was closed.")
            break
        transcript.append(("BORROWER", caller))
        agent_text = await agent.respond(caller)
        transcript.append(("AGENT", agent_text))

    agent.state.save()  # exercise the same write path as the live bot
    record = agent.state.outcome.model_dump(mode="json") if agent.state.outcome else None
    scored = rule_checks(scenario, borrower, agent.events, record, agent.state.turns)
    text = "\n".join(f"{who}: {line}" for who, line in transcript)
    llm_secs = [e["secs"] for e in agent.events if e["type"] == "llm"]

    result = {
        "scenario": scenario["id"],
        "run": run_idx,
        "record": record,
        "expected": scenario.get("expected", {}),
        **scored,
        "llm_secs": llm_secs,
        "cache_reads": [e["cache_read"] for e in agent.events if e["type"] == "llm"],
        "llm_calls": [{"blocks": e.get("blocks"), "tools": e.get("tools")} for e in agent.events if e["type"] == "llm"],
        "tool_errors": [e["error"] for e in agent.events if e["type"] == "tool_error"],
        "transcript": text,
    }
    if use_judge:
        result["judge"] = (await judge(client, scenario, text)).model_dump()
    return result


def gate(results: list[dict], min_pass_rate: float) -> tuple[bool, list[str]]:
    """The CI verdict. Two rules: the pass rate must reach `min_pass_rate`, and, whatever the
    rate, no run may disclose loan details before verification or have a judge-flagged guardrail
    violation. One leak in seven runs is a failed build, not 86%. (With --no-judge only the
    disclosure rule can fire.)"""
    messages = []
    for r in results:
        where = f"{r['scenario']} run {r['run']}"
        if r["checks"].get("no_disclosure_before_verification") is False:
            leaked = (r.get("leaks") or ["?"])[0]
            messages.append(f"ZERO TOLERANCE: {where} disclosed loan details before verification: \"{leaked[:120]}\"")
        if (r.get("judge") or {}).get("guardrail_violation"):
            quote = ((r["judge"].get("violations") or ["?"])[0])[:120]
            messages.append(f"ZERO TOLERANCE: {where} has a guardrail violation (judge): \"{quote}\"")
    rate = sum(all(r["checks"].values()) for r in results) / max(len(results), 1)
    if rate < min_pass_rate:
        messages.append(f"FAIL: pass rate {rate:.0%} is below the required {min_pass_rate:.0%}")
    ok = not messages
    if ok:
        messages.append(f"OK: pass rate {rate:.0%} meets the required {min_pass_rate:.0%}, "
                        "no disclosures or guardrail violations")
    return ok, messages


def summarise(results: list[dict]) -> str:
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)

    def rate(rs, key):
        vals = [r["checks"][key] for r in rs if key in r["checks"]]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"

    passed = sum(all(r["checks"].values()) for r in results)
    lines = [
        f"Prompt version {prompt_version()}: {passed}/{len(results)} runs pass ({passed / max(len(results), 1):.0%}).",
        "",
        "| scenario | outcome | disclosure-safe | closed | ptp date | escalation | judge tone | judge violations | avg turns |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sid, rs in by_scenario.items():
        judged = [r["judge"] for r in rs if "judge" in r]
        tone = f"{statistics.mean(j['tone'] for j in judged):.1f}" if judged else "-"
        violations = str(sum(j["guardrail_violation"] for j in judged)) if judged else "-"
        lines.append(
            f"| {sid} | {rate(rs, 'outcome_correct')} | {rate(rs, 'no_disclosure_before_verification')} "
            f"| {rate(rs, 'call_closed')} | {rate(rs, 'promise_to_pay_date_correct')} "
            f"| {rate(rs, 'escalate_to_human_correct')} | {tone} | {violations} "
            f"| {statistics.mean(r['turns'] for r in rs):.1f} |"
        )

    llm = sorted(s for r in results for s in r["llm_secs"])
    if llm:
        p50 = statistics.median(llm)
        p90 = llm[int(0.9 * (len(llm) - 1))]
        lines += ["", f"Text-agent LLM response time (full response, not TTFT): median {p50:.2f}s, p90 {p90:.2f}s over {len(llm)} calls"]

    raw = sum(len(r.get("raw_numbers", [])) for r in results)
    reads = [e for r in results for e in r.get("cache_reads", [])]
    lines.append(f"Agent lines with raw digits/₹ (formatter needed): {raw}")
    vt = sum(r.get("goodbye_via_tool", 0) for r in results)
    lines.append(f"Calls whose goodbye was spoken from end_call.goodbye: {vt}/{len(results)}")
    te = sum(len(r.get("tool_errors", [])) for r in results)
    if te:
        lines.append(f"Malformed tool calls rejected by the provider and retried: {te}")
    if reads and any(reads):
        lines.append(f"Prompt cache hit rate: {sum(1 for x in reads if x > 0)}/{len(reads)} LLM calls")

    failures = [r for r in results if not all(r["checks"].values())]
    lines += ["", f"## Failing runs ({len(failures)}/{len(results)})", ""]
    for r in failures:
        bad = [k for k, v in r["checks"].items() if not v]
        lines.append(f"- **{r['scenario']}** run {r['run']}: {', '.join(bad)}")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scenario", action="append", help="Scenario id (repeatable). Default: all.")
    parser.add_argument("-n", "--runs", type=int, default=3)
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--agent-model", help="Override AGENT_MODEL (e.g. claude-haiku-4-5, groq:openai/gpt-oss-120b, cerebras:gpt-oss-120b)")
    parser.add_argument("--thinking", choices=["adaptive", "between_tools"], help="Override AGENT_THINKING")
    parser.add_argument("--tag", default="", help="Suffix for the results folder name")
    parser.add_argument("--subset", help="Only scenarios tagged with this subset in scenarios.yaml (e.g. ci)")
    parser.add_argument("--min-pass-rate", type=float,
                        help="CI gate: exit 1 if fewer than this share of runs pass, or if ANY run discloses "
                             "loan details before verification or has a judge-flagged guardrail violation")
    args = parser.parse_args()

    scenarios = yaml.safe_load(SCENARIOS_PATH.read_text())["scenarios"]
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] in args.scenario]
    if args.subset:
        scenarios = [s for s in scenarios if args.subset in s.get("subsets", [])]

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(s, i):
        async with sem:
            try:
                r = await run_one(client, s, i, not args.no_judge, args.agent_model, args.thinking)
            except Exception as e:
                r = {"scenario": s["id"], "run": i, "error": repr(e), "checks": {"run_completed": False}, "turns": 0, "llm_secs": []}
            ok = all(r["checks"].values())
            print(f"{'PASS' if ok else 'FAIL'} {s['id']} #{i}")
            return r

    results = await asyncio.gather(*(guarded(s, i) for s in scenarios for i in range(args.runs)))

    out = ROOT / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True)
    Path(out / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    summary = summarise(results)
    (out / "summary.md").write_text(summary + "\n")
    print("\n" + summary + f"\n\nSaved to {out}")
    if args.min_pass_rate is not None:
        ok, messages = gate(results, args.min_pass_rate)
        print("\n" + "\n".join(messages))
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
