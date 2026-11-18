"""An LLM that plays the borrower, driven by a scenario persona."""

import anthropic

from agent.config import CONFIG

HANGUP = "<hangup>"

SIMULATOR_PROMPT = """\
You are role-playing a person in India who has just answered a phone call. Stay in character for the whole call.

# Who you are
{persona}

# What you do on this call
{behaviour}

# Facts you know (only share what a real person would, when asked)
{facts}

# Rules
- Reply with only the words you say out loud: one short, natural phone utterance. No stage directions, no quotes, no narration.
- Talk like a real person on the phone: casual, sometimes brief ("haan", "okay", "yeah"), occasionally a little messy.
- Do not make the agent's job easy: only volunteer information your character would.
- When the call is clearly over (they said goodbye, or you want to hang up), reply with exactly {hangup}.
"""


class BorrowerSimulator:
    def __init__(self, client: anthropic.AsyncAnthropic, scenario: dict, facts: str):
        self.client = client
        self.system = SIMULATOR_PROMPT.format(
            persona=scenario["persona"].strip(),
            behaviour=scenario["behaviour"].strip(),
            facts=facts,
            hangup=HANGUP,
        )
        # From the simulator's point of view the agent is the "user".
        self.messages: list[dict] = []

    async def reply(self, agent_text: str) -> str:
        self.messages.append({"role": "user", "content": agent_text or "(silence)"})
        response = await self.client.beta.messages.create(
            model=CONFIG.simulator_model,
            max_tokens=1024,
            system=self.system,
            messages=self.messages,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        text = " ".join(b.text for b in response.content if b.type == "text").strip()
        self.messages.append({"role": "assistant", "content": response.content})
        return text or HANGUP
