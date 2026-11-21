"""Text-level twin of the voice agent, for automated evals.

Same system prompt, same tool specs, same ``run_tool`` guardrails and the same
``CallState`` as agent/bot.py; only the audio path (VAD, STT, TTS) is skipped.
Spoken output is passed through the same speech formatter so checks see what
the caller would hear.
"""

import time
from pathlib import Path

import anthropic

from agent import templates
from agent.borrowers import Borrower
from agent.call_state import CallState
from agent.config import CONFIG, model_supports_effort
from agent.prompts import build_system_prompt, first_message_instruction
from agent.speech import to_spoken
from agent.tools import closing_line, run_tool, tool_specs


def anthropic_tools() -> list[dict]:
    return [
        {"name": s["name"], "description": s["description"],
         "input_schema": {"type": "object", "properties": s["properties"], "required": s["required"]}}
        for s in tool_specs()
    ]

MAX_TOOL_ROUNDS = 4


class TextAgent:
    def __init__(self, client: anthropic.AsyncAnthropic, borrower: Borrower, history_db: Path,
                 model: str | None = None, thinking: str | None = None):
        self.client = client
        self.model = model or CONFIG.agent_model
        self.thinking = thinking or CONFIG.agent_thinking
        # Simulated calls are audited next to their throwaway history, never in the real log.
        self.state = CallState(borrower=borrower, language=borrower.preferred_language, history_db=history_db,
                               audit_path=Path(history_db).parent / "audit.jsonl")
        self.system = build_system_prompt(borrower)
        self.messages: list[dict] = []
        self.events: list[dict] = []  # tool calls and timings, for the metrics

    async def start(self) -> str:
        if CONFIG.templated_turns:  # same as the live bot: templated greeting, no LLM call
            b = self.state.borrower
            line = templates.greeting(b)
            self.messages += [
                {"role": "user", "content": templates.CALL_CONNECTED_NOTE.format(name=b.name)},
                {"role": "assistant", "content": line},
            ]
            self.events.append({"type": "say", "text": line, "raw": line, "verified": False, "template": True})
            return line
        return await self.respond(f"[Call connected] {first_message_instruction(self.state.borrower)}", count_turn=False)

    async def respond(self, caller_text: str, *, count_turn: bool = True) -> str:
        if count_turn:
            self.state.turns += 1
        self.messages.append({"role": "user", "content": caller_text})
        spoken: list[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            t0 = time.perf_counter()
            extra = {"thinking": {"type": "between_tools"}} if self.thinking == "between_tools" else {}
            system: str | list = self.system
            if CONFIG.prompt_caching:
                # Same caching as the live bot: system prompt, plus the conversation so far.
                system = [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}]
                extra["cache_control"] = {"type": "ephemeral"}
            if model_supports_effort(self.model):
                extra["output_config"] = {"effort": CONFIG.agent_effort}
                extra["betas"] = ["server-side-fallback-2026-07-01"]
                extra["fallbacks"] = "default"
            else:
                extra.pop("thinking", None)  # between_tools is Sonnet 5.5 only
            response = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
                tools=anthropic_tools(),
                messages=self.messages,
                **extra,
            )
            u = response.usage
            self.events.append({"type": "llm", "secs": round(time.perf_counter() - t0, 3), "stop": response.stop_reason,
                                "cache_read": u.cache_read_input_tokens or 0,
                                "blocks": [b.type for b in response.content],
                                "tools": [b.name for b in response.content if b.type == "tool_use"]})
            # Keep the full content (thinking + tool_use blocks) so the next request is valid.
            self.messages.append({"role": "assistant", "content": response.content})

            texts = [b.text for b in response.content if b.type == "text" and b.text.strip()]
            spoken += texts
            for text in texts:
                # Verification status at the moment this was said, for the disclosure check.
                self.events.append({"type": "say", "text": to_spoken(text), "raw": text, "verified": self.state.verified})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                break

            results = []
            template_line = None
            for block in tool_uses:
                result = await run_tool(self.state, block.name, dict(block.input))
                if block.name == "verify_identity" and CONFIG.templated_turns and len(tool_uses) == 1:
                    template_line = templates.reminder(self.state.borrower, result, self.state.language)
                if block.name == "end_call":  # same as the live bot: speak the goodbye field
                    if line := closing_line(dict(block.input), " ".join(spoken), self.state.language):
                        spoken.append(line)
                        self.events.append({"type": "say", "text": to_spoken(line), "raw": line,
                                            "verified": self.state.verified, "from_tool": True})
                self.events.append({"type": "tool", "name": block.name, "input": dict(block.input), "result": result, "verified_after": self.state.verified})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(result)})
            self.messages.append({"role": "user", "content": results})
            if template_line:  # same as the live bot: templated reminder, no second LLM pass
                self.messages.append({"role": "assistant", "content": template_line})
                spoken.append(template_line)
                self.events.append({"type": "say", "text": template_line, "raw": template_line,
                                    "verified": self.state.verified, "template": True})
                break
            if self.state.ended:
                break

        return to_spoken(" ".join(spoken))


# Open-weight model hosts with OpenAI-compatible APIs: prefix -> (base URL, key variable).
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
}
GROQ_BASE_URL = PROVIDERS["groq"][0]


def openai_tools() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": s["name"], "description": s["description"],
            "parameters": {"type": "object", "properties": s["properties"], "required": s["required"]}}}
        for s in tool_specs()
    ]


class OpenAICompatTextAgent(TextAgent):
    """Same agent on an open-weight model hosted by Groq or Cerebras (OpenAI-compatible API).

    Same prompt, tools, guardrails, templates and goodbye handling as TextAgent; only the
    wire format differs. Select with --agent-model <provider>:<model>,
    e.g. groq:openai/gpt-oss-120b or cerebras:gpt-oss-120b.
    """

    def __init__(self, borrower: Borrower, history_db: Path, model: str, reasoning_effort: str | None = "low"):
        import os

        from openai import AsyncOpenAI

        provider, _, name = model.partition(":")
        base_url, key_var = PROVIDERS[provider]
        super().__init__(None, borrower, history_db, model=name)
        self.client = AsyncOpenAI(api_key=os.environ[key_var], base_url=base_url)
        self.reasoning_effort = reasoning_effort

    async def respond(self, caller_text: str, *, count_turn: bool = True) -> str:
        import json

        from openai import BadRequestError

        if count_turn:
            self.state.turns += 1
        self.messages.append({"role": "user", "content": caller_text})
        spoken: list[str] = []

        for _ in range(MAX_TOOL_ROUNDS):
            t0 = time.perf_counter()
            extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
            # The host may validate tool calls server-side and rejects malformed ones (unknown tool,
            # null for a string field). Retry like a production client would, and count each
            # rejection: it is both an accuracy signal and extra latency.
            for attempt in range(3):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        max_tokens=2048,
                        messages=[{"role": "system", "content": self.system}, *self.messages],
                        tools=openai_tools(),
                        **extra,
                    )
                    break
                except BadRequestError as e:
                    msg = str(e).lower()
                    if ("tool call validation failed" not in msg and "failed to parse tool call" not in msg) or attempt == 2:
                        raise
                    self.events.append({"type": "tool_error", "error": str(e)[:200]})
            msg = response.choices[0].message
            calls = msg.tool_calls or []
            self.events.append({"type": "llm", "secs": round(time.perf_counter() - t0, 3),
                                "stop": response.choices[0].finish_reason, "cache_read": 0,
                                "blocks": (["text"] if (msg.content or "").strip() else []) + ["tool_use"] * len(calls),
                                "tools": [c.function.name for c in calls]})
            entry = {"role": "assistant", "content": msg.content or ""}
            if calls:
                entry["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                       for c in calls]
            self.messages.append(entry)

            if text := (msg.content or "").strip():
                spoken.append(text)
                self.events.append({"type": "say", "text": to_spoken(text), "raw": text, "verified": self.state.verified})
            if not calls:
                break

            template_line = None
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await run_tool(self.state, c.function.name, args)
                if c.function.name == "verify_identity" and CONFIG.templated_turns and len(calls) == 1:
                    template_line = templates.reminder(self.state.borrower, result, self.state.language)
                if c.function.name == "end_call":
                    if line := closing_line(args, " ".join(spoken), self.state.language):
                        spoken.append(line)
                        self.events.append({"type": "say", "text": to_spoken(line), "raw": line,
                                            "verified": self.state.verified, "from_tool": True})
                self.events.append({"type": "tool", "name": c.function.name, "input": args, "result": result,
                                    "verified_after": self.state.verified})
                self.messages.append({"role": "tool", "tool_call_id": c.id, "content": str(result)})
            if template_line:
                self.messages.append({"role": "assistant", "content": template_line})
                spoken.append(template_line)
                self.events.append({"type": "say", "text": template_line, "raw": template_line,
                                    "verified": self.state.verified, "template": True})
                break
            if self.state.ended:
                break

        return to_spoken(" ".join(spoken))
