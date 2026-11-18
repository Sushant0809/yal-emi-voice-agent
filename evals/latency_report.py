"""Layer-by-layer latency of live calls, from the per-turn log the bot writes.

    uv run python -m evals.latency_report                       # every logged call
    uv run python -m evals.latency_report --since 20261118T1722 # calls from this call_id on
    uv run python -m evals.latency_report --last 5              # the last 5 calls

Each turn runs from the moment the caller stops speaking until Asha's first audio plays,
as measured by Pipecat's UserBotLatencyObserver (outcomes/latency.jsonl).
"""

import argparse
import json
import statistics

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
    s = sorted(values)
    return s[int(q * (len(s) - 1))]


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


def report(turns: list[dict]) -> str:
    calls = {t["call_id"] for t in turns}
    lines = [f"{len(turns)} turns across {len(calls)} calls", "",
             "| Layer | Turns | Median | p90 | Min | Max |", "|---|---|---|---|---|---|"]
    for key, label in LAYERS:
        v = [c["duration_secs"] for t in turns for c in t["contributions"] if c["key"] == key]
        if v:
            lines.append(f"| {label} | {len(v)} | {statistics.median(v):.2f}s | {_pct(v, 0.9):.2f}s "
                         f"| {min(v):.2f}s | {max(v):.2f}s |")
    total = [t["total_secs"] for t in turns]
    lines.append(f"| **Total, voice to voice** | {len(total)} | **{statistics.median(total):.2f}s** "
                 f"| {_pct(total, 0.9):.2f}s | {min(total):.2f}s | {max(total):.2f}s |")
    tool = [t["total_secs"] for t in turns if any(c["key"] == "llm_tool_call" for c in t["contributions"])]
    direct = [t["total_secs"] for t in turns if not any(c["key"] == "llm_tool_call" for c in t["contributions"])]
    lines.append("")
    if direct:
        lines.append(f"Replies answered directly by the LLM: median {statistics.median(direct):.2f}s ({len(direct)} turns)")
    if tool:
        lines.append(f"Turns with a tool call (filler covers the wait): median {statistics.median(tool):.2f}s ({len(tool)} turns)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", help="Only calls whose call_id is >= this (call ids sort by time)")
    parser.add_argument("--last", type=int, help="Only the last N calls")
    args = parser.parse_args()
    turns = load_turns(args.since, args.last)
    print(report(turns) if turns else "No turns logged yet. Make a call first.")


if __name__ == "__main__":
    main()
