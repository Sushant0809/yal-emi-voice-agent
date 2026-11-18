"""Time-to-first-token benchmark for candidate agent LLM settings.

Sends the same realistic turns (real system prompt + tools, warm prompt cache) to
each configuration and reports when the first *spoken* text token arrives, which is
what the caller waits for. Thinking time counts against it.

    uv run python -m evals.bench_llm -n 5
"""

import argparse
import asyncio
import statistics
import time

import anthropic

from agent.borrowers import get_borrower
from agent.prompts import build_system_prompt
from evals.text_agent import anthropic_tools

CONFIGS = [
    ("haiku-4.5 (no thinking)", "claude-haiku-4-5", None, None),
    ("sonnet-5.5 low adaptive", "claude-sonnet-5-5", "low", None),
    ("sonnet-5.5 low between_tools", "claude-sonnet-5-5", "low", {"type": "between_tools"}),
]

# Representative turns: the greeting, a plain reply, and the turn after a tool result
# (the slowest kind, since the model has to read the loan details and phrase them).
TURNS = {
    "greeting": [
        {"role": "user", "content": "[Call connected] Greet the person and ask if you are speaking with Rahul Sharma. Keep it to one sentence."},
    ],
    "after_name": [
        {"role": "user", "content": "[Call connected] Greet the person and ask if you are speaking with Rahul Sharma. Keep it to one sentence."},
        {"role": "assistant", "content": "Hi, this is Asha from Lumen Finance. Am I speaking with Rahul Sharma?"},
        {"role": "user", "content": "Haan, yes, Rahul here. What's this about?"},
    ],
    "after_verify_tool": [
        {"role": "user", "content": "[Call connected] Greet the person and ask if you are speaking with Rahul Sharma. Keep it to one sentence."},
        {"role": "assistant", "content": "Hi, this is Asha from Lumen Finance. Am I speaking with Rahul Sharma?"},
        {"role": "user", "content": "Yes, speaking."},
        {"role": "assistant", "content": "Thanks, Rahul. Could you confirm your date of birth or the last four digits of your registered mobile?"},
        {"role": "user", "content": "Fourteenth March 1990."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_bench1", "name": "verify_identity", "input": {"dob": "1990-03-14"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_bench1", "content": str({
            "verified": True,
            "loan": {"loan_type": "Personal Loan", "emi_amount_inr": 12500, "due_date": "2026-12-05", "days_overdue": 0,
                     "overdue_amount_inr": 0, "emi_amount_spoken": "twelve thousand five hundred rupees",
                     "due_date_spoken": "the fifth of December"},
        })}]},
    ],
    # Hard turns copied from a live call, where adaptive thinking kicked in (0.8-1.8s).
    "after_failed_verify": [
        {"role": "user", "content": "[Call connected] Greet the person and ask if you are speaking with Rahul Sharma. Keep it to one sentence."},
        {"role": "assistant", "content": "Hello, this is Asha calling from Lumen Finance. Am I speaking with Rahul Sharma?"},
        {"role": "user", "content": "Yes, I'm Rahul."},
        {"role": "assistant", "content": "Thanks, Rahul. Could you confirm your date of birth or the last four digits of your registered mobile?"},
        {"role": "user", "content": "My registered mobile number is eight five six six."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_bench2", "name": "verify_identity", "input": {"phone_last4": "8566"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_bench2", "content": "{'verified': False, 'attempts_remaining': 1}"}]},
    ],
    "changes_date": [
        {"role": "user", "content": "[Call connected] Greet the person and ask if you are speaking with Rahul Sharma. Keep it to one sentence."},
        {"role": "assistant", "content": "Hello, this is Asha calling from Lumen Finance. Am I speaking with Rahul Sharma?"},
        {"role": "user", "content": "Yes."},
        {"role": "assistant", "content": "Your EMI of twelve thousand five hundred rupees is due on the fifth of December. Will you be able to pay it on time?"},
        {"role": "user", "content": "Yes, I want to pay on the fifth. And... but due to some technical reason I can't pay on the fifth, I will be paying on the seventh."},
    ],
}


async def ttft(client, model, effort, thinking, system, messages) -> tuple[float, float, int]:
    kwargs = {"thinking": thinking} if thinking else {}
    if effort:
        kwargs["output_config"] = {"effort": effort}
    t0 = time.perf_counter()
    first = None
    async with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        tools=anthropic_tools(),
        messages=messages,
        **kwargs,
    ) as stream:
        async for _ in stream.text_stream:
            if first is None:
                first = time.perf_counter() - t0
        final = await stream.get_final_message()
    total = time.perf_counter() - t0
    return (first if first is not None else total), total, final.usage.cache_read_input_tokens or 0


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--reps", type=int, default=5)
    args = parser.parse_args()

    client = anthropic.AsyncAnthropic()
    system = build_system_prompt(get_borrower("LF-1001"))

    print(f"{'config':32s} {'turn':18s} {'TTFT p50':>9s} {'TTFT p90':>9s} {'total p50':>10s} {'cache hit':>9s}")
    for label, model, effort, thinking in CONFIGS:  # warm each model's cache (caches are per model)
        await ttft(client, model, effort, thinking, system, TURNS["greeting"])
    # Interleave configs within each repetition so network drift over time can't favour one config.
    results = {(c[0], t): [] for c in CONFIGS for t in TURNS}
    for _ in range(args.reps):
        for turn, messages in TURNS.items():
            for label, model, effort, thinking in CONFIGS:
                results[(label, turn)].append(await ttft(client, model, effort, thinking, system, messages))
    for label, *_ in CONFIGS:
        for turn in TURNS:
            runs = results[(label, turn)]
            firsts = sorted(r[0] for r in runs)
            p90 = firsts[min(len(firsts) - 1, int(round(0.9 * (len(firsts) - 1))))]
            hit = sum(1 for r in runs if r[2] > 0)
            print(f"{label:32s} {turn:18s} {statistics.median(firsts):8.2f}s {p90:8.2f}s "
                  f"{statistics.median(r[1] for r in runs):9.2f}s {hit:>5d}/{len(runs)}")

if __name__ == "__main__":
    asyncio.run(main())
