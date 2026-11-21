"""Save a readable transcript of every live call.

    outcomes/transcripts/<call_id>.txt    AGENT / CALLER lines, tool calls inline
    outcomes/transcripts/<call_id>.json   the raw LLM context
    outcomes/recordings/<call_id>.wav     the call audio (see agent/recording.py)

    uv run python -m agent.transcripts            # list saved calls
    uv run python -m agent.transcripts <call_id>  # print one
"""

import json
import sys
from pathlib import Path

from agent.config import CONFIG

TRANSCRIPTS_DIR = CONFIG.outcomes_dir / "transcripts"


def _blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)]


def _plain(msg) -> dict:
    """Pipecat wraps provider-specific messages (e.g. Anthropic thinking blocks) in
    LLMSpecificMessage(llm=..., message=...); unwrap to the underlying dict."""
    inner = getattr(msg, "message", msg)
    return inner if isinstance(inner, dict) else {"role": "unknown", "content": str(inner)}


def render(messages: list) -> str:
    lines = []
    for i, msg in enumerate(map(_plain, messages)):
        role = msg.get("role")
        for block in _blocks(msg.get("content", "")):
            kind = block.get("type")
            if role == "tool":
                lines.append(f"  [result] {str(block.get('text', block.get('content', '')))[:300]}")
            elif kind == "text" and block.get("text", "").strip():
                # The first user message is our call-connected instruction, not the caller.
                who = "AGENT " if role == "assistant" else ("SYSTEM" if i == 0 else "CALLER")
                lines.append(f"{who}: {block['text'].strip()}")
            elif kind == "tool_use":
                lines.append(f"  [tool] {block.get('name')}({json.dumps(block.get('input', {}), ensure_ascii=False)})")
            elif kind == "tool_result":
                lines.append(f"  [result] {str(block.get('content'))[:300]}")
    return "\n".join(lines) + "\n"


def save(call_id: str, messages: list, outcome: dict | None) -> Path:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    messages = [_plain(m) for m in messages]
    text = render(messages)
    if outcome:
        text += f"\nOUTCOME: {json.dumps(outcome, ensure_ascii=False)}\n"
    path = TRANSCRIPTS_DIR / f"{call_id}.txt"
    path.write_text(text)
    (TRANSCRIPTS_DIR / f"{call_id}.json").write_text(
        json.dumps({"call_id": call_id, "outcome": outcome, "messages": messages}, ensure_ascii=False, default=str, indent=1)
    )
    return path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from agent import audit

        audit.record(audit.cli_actor(), "view_transcript", call_id=sys.argv[1])
        print((TRANSCRIPTS_DIR / f"{sys.argv[1]}.txt").read_text())
    else:
        for p in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
            print(p.stem)
