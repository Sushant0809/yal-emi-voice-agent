"""Layer-by-layer latency of live calls, from the per-turn log the bot writes.

    uv run python -m evals.latency_report                       # every logged call
    uv run python -m evals.latency_report --since 20261118T1722 # calls from this call_id on
    uv run python -m evals.latency_report --last 5              # the last 5 calls

Each turn runs from the moment the caller stops speaking until Asha's first audio plays,
as measured by Pipecat's UserBotLatencyObserver (outcomes/latency.jsonl). Percentiles are
nearest-rank. The interruption rate (outcomes/turns.jsonl) is the share of Asha's turns the
caller cut off by speaking; the turn still open when the call hangs up isn't counted.
"""

import argparse
import json
import math

from agent.config import CONFIG

LAYERS = [
    ("endpointing_wait", "Endpointing wait (VAD)"),
    ("transcription", "Speech-to-text (final transcript)"),
    ("turn_detection", "Turn detection"),
    ("llm_inference", "LLM time to first token"),
    ("llm_tool_call", "LLM tool call (tool turns only)"),
    ("function_handler", "Tool handler (our code)"),
    ("sentence_aggregation", "Sentence wait before TTS"),
    ("speech_synthesis", "Text-to-speech first audio"),
    ("pipeline", "Pipeline / transport"),
]


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def _stats(v: list[float]) -> str:
    return (f"{len(v)} | {_pct(v, 0.5):.2f}s | {_pct(v, 0.9):.2f}s | {_pct(v, 0.95):.2f}s "
            f"| {min(v):.2f}s | {max(v):.2f}s")


def load_turns(since: str | None, last: int | None) -> list[dict]:
    path = CONFIG.outcomes_dir / "latency.jsonl"
    turns = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    turns = [t for t in turns if t.get("measured_from") == "user_silence"]
    if since:
        turns = [t for t in turns if t["call_id"] >= since]
    if last:
        keep = sorted({t["call_id"] for t in turns})[-last:]
        turns = [t for t in turns if t["call_id"] in keep]
    return turns


def load_turn_log(call_ids: set[str]) -> list[dict]:
    path = CONFIG.outcomes_dir / "turns.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["call_id"] in call_ids]


def interruptions(turn_log: list[dict]) -> str:
    counted = [r for r in turn_log if not r.get("call_ended")]
    if not counted:
        return "Interruption rate: no turn data for these calls yet (logged from 21 Nov 2026 on)."
    cut = sum(r["was_interrupted"] for r in counted)
    calls = {r["call_id"] for r in counted}
    with_cut = {r["call_id"] for r in counted if r["was_interrupted"]}
    return (f"Interruption rate: {cut}/{len(counted)} turns ({cut / len(counted):.0%}) were cut off by the caller; "
            f"{len(with_cut)} of {len(calls)} calls had at least one.")


def report(turns: list[dict], turn_log: list[dict] | None = None) -> str:
    calls = {t["call_id"] for t in turns}
    lines = [f"{len(turns)} turns across {len(calls)} calls", "",
             "| Layer | Turns | p50 | p90 | p95 | Min | Max |", "|---|---|---|---|---|---|---|"]
    for key, label in LAYERS:
        v = [c["duration_secs"] for t in turns for c in t["contributions"] if c["key"] == key]
        if v:
            lines.append(f"| {label} | {_stats(v)} |")
    total = [t["total_secs"] for t in turns]
    lines.append(f"| **Total, voice to voice** | {_stats(total)} |")
    tool = [t["total_secs"] for t in turns if any(c["key"] == "llm_tool_call" for c in t["contributions"])]
    direct = [t["total_secs"] for t in turns if not any(c["key"] == "llm_tool_call" for c in t["contributions"])]
    lines.append("")
    if direct:
        lines.append(f"Replies answered directly by the LLM: p50 {_pct(direct, 0.5):.2f}s, p95 {_pct(direct, 0.95):.2f}s ({len(direct)} turns)")
    if tool:
        lines.append(f"Turns with a tool call (filler covers the wait): p50 {_pct(tool, 0.5):.2f}s, p95 {_pct(tool, 0.95):.2f}s ({len(tool)} turns)")
    if turn_log is not None:
        lines.append(interruptions(turn_log))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", help="Only calls whose call_id is >= this (call ids sort by time)")
    parser.add_argument("--last", type=int, help="Only the last N calls")
    args = parser.parse_args()
    turns = load_turns(args.since, args.last)
    if not turns:
        print("No turns logged yet. Make a call first.")
        return
    print(report(turns, load_turn_log({t["call_id"] for t in turns})))


if __name__ == "__main__":
    main()
