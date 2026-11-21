# Lumen Finance: Conversational EMI Reminder Voice Agent

An outbound voice agent that reminds borrowers about upcoming or overdue EMIs. It verifies who it's talking to, states the amount clearly, finds out what the borrower intends to do, offers a payment link, and records a structured outcome. It's built on [Pipecat](https://github.com/pipecat-ai/pipecat) and runs in the browser on a laptop.

> Status: working end to end. Full eval suite (19 scenarios × 2) passes 38/38 with 0 guardrail violations, and the payment-link failure scenario passes 3/3; median voice-to-voice latency 2.25s on live calls, p95 4.72s (see the layer-by-layer table).

## Quick start

```bash
uv sync
cp .env.example .env          # add ANTHROPIC / DEEPGRAM / CARTESIA keys and a Cartesia voice id
uv run python -m agent.bot    # open http://localhost:7860 and click Connect
```

Calls are only placed between 8am and 7pm IST (`CALLING_HOURS`); outside that window the bot logs a warning and hangs up without speaking. Set `ENFORCE_CALLING_HOURS=false` to test in the evening.

Pick who Asha calls at **http://localhost:7860/borrowers?token=…** (the token is `ADMIN_TOKEN` in `.env`, or a random one printed in the log at start-up): it shows each borrower's verification details and past calls, can clear their history, and links to the call page (no restart needed). The selection is stored as `DEFAULT_BORROWER_ID` in `.env`. Every call is recorded to `outcomes/recordings/<call_id>.wav` (stereo: caller left, Asha right) and transcribed to `outcomes/transcripts/<call_id>.txt`. Every call's outcome is saved to the call-history database (`outcomes/history.db`), and per-turn latency goes to `outcomes/latency.jsonl`. To demo cross-call memory, run `uv run python -m agent.history seed`, then call `LF-1006`, who broke a promise to pay on the 10th.

```bash
uv run pytest                                   # unit tests: formatter, verification, guardrails
uv run python -m evals.run_evals -n 3           # every scenario x3, rule checks + LLM judge
uv run python -m evals.run_evals -s hardship -n 5 --no-judge
uv run python -m evals.run_evals -n 2 --agent-model cerebras:qwen-3.8-27b   # same suite, another model
uv run python -m evals.latency_report --last 5  # layer-by-layer latency (p50/p95) and interruption rate
uv run python -m evals.online                   # score today's real calls; list calls for human review
uv run python -m evals.wer                      # STT accuracy on digits, dates and amounts (en / hi / Hinglish)
```

## Architecture

```
Browser mic --WebRTC--> Silero VAD --> Deepgram Flux multilingual (STT + turn detection + eager end-of-turn)
                                                              |
                                                              v
Browser speaker <--WebRTC-- Cartesia Sonic <-- speech formatter <-- Claude (tools from the registry)
                                                                         |
                                                                   CallState (code-owned)
```

| Concern | How it's handled |
|---|---|
| Know when the caller has finished | **Deepgram Flux (multilingual, includes Hindi)** does transcription and turn detection in one model. Fallback with `STT_ENGINE=nova`: Nova-3 plus **Smart Turn v3** (an audio end-of-turn model) with a 1.2s silence fallback |
| Start answering before the turn is confirmed | Flux emits an *eager* end-of-turn prediction. Pipecat starts the LLM on it and holds the reply until the turn is confirmed, discarding it if the caller keeps talking (`EagerUserTurnStrategies`) |
| Barge-in | Pipecat cancels in-flight LLM and TTS output when the user starts speaking |
| Low latency | Streaming STT; whole sentences sent to TTS (`TTS_TEXT_MODE=sentence`; token streaming was 0.3s faster but stretched a lone first word like "Thank"); prompt caching plus a `max_tokens=0` pre-warm while the call connects; one Anthropic client shared across calls; VAD and Smart Turn preloaded at start-up. `UserBotLatencyObserver` logs a per-stage breakdown every turn |
| No dead air during tools | Each tool can declare a filler line in the phrase pack. `on_function_calls_started` speaks it in the caller's language while the tool runs: "Thanks, one moment" (`verify_identity`), "Let me just check that for you" (`check_payment_status`), "Sure, I'm sending that to you now" (`send_payment_link`). After verification, the reminder is spoken from a template, so no second LLM pass is needed |
| Silence | `on_user_turn_idle`: "are you still there?" twice, then a graceful goodbye and a `no_response` outcome |
| Numbers spoken like a person | `agent/speech.py` TTS transform: ₹12,500 becomes "twelve thousand five hundred rupees", 2026-12-05 becomes "the fifth of December". Indian lakh/crore system |
| Security and compliance | Identity checks take one factor at a time and are logged, so a borrower locks out after 3 failures across calls in 24h (not just 2 per call). Calling hours (8am–7pm IST) enforced in code. One payment link per call, keyed by call id. Promise dates are validated before they are stored. DOBs, phone digits and four-digit groups are masked in every log line (`agent/redact.py`). The borrower page needs a token, and all its data is escaped |
| Tool failures | `run_tool` never raises. Failures return a message the model apologises around. `PAYMENT_LINK_FAILURE_RATE` injects failures |
| Background noise | **RNNoise** (open-source, local CPU, ~1.5ms per 20ms frame, about 56 dB reduction on stationary noise in a unit test) cleans caller audio before Silero VAD (confidence ≥ 0.7, volume ≥ 0.6) and Flux see it. Noise that yields no transcript is ignored unless it interrupted Asha. Garbled speech gets "sorry, could you say that again?". `NOISE_FILTER=off` for A/B tests |
| Provider drop-outs | `agent/recovery.py`, driven by real signals rather than timers. A reply that produced no audio is held, then re-spoken with "Sorry, the line broke up for a second" when Cartesia actually reconnects (`on_connected`). An STT drop makes Asha ask the caller to repeat. It counts as recovered only when the audio plays. If the caller speaks first, the apology is dropped and the LLM answers normally. One apology per outage, in the caller's language. Incidents go to `outcomes/incidents.jsonl` and `connection_incidents` in the outcome |

## Key design decisions

1. **Verification is enforced in code, not only in the prompt.** The system prompt contains only the borrower's name. Loan details enter the context *only* through a successful `verify_identity` result, so a family member, a wrong number or a prompt-injection attempt has nothing to extract. Verification locks after two failures in a call, and after three across calls in 24 hours (each check is stored in `verification_attempts`), so calling back to guess again doesn't work. `verified` is set only once the whole result has been built.
2. **Cross-call memory sits behind the same gate.** Past calls are loan information, so `verify_identity` only returns them after verification succeeds (see Memory below).
3. **Escalation for hardship and disputes is forced in code.** `CallState.finalize` sets `escalate_to_human=True` for those outcomes, whatever the model passes.
4. **One brain, two bodies.** `agent/tools.py`, `agent/prompts.py` and `agent/call_state.py` don't depend on Pipecat. The live bot (`agent/bot.py`) and the eval agent (`evals/text_agent.py`) share them, so the evals test the same guardrails the caller hits.
5. **Pre-rendered spoken amounts.** `verify_identity` returns `emi_amount_spoken` and `due_date_spoken` alongside the raw values, so the model rarely has to convert numbers to words. The TTS transform catches the rest.
6. **Goals, not a script; tools as a registry.** The system prompt describes what a good call achieves, the rules that always hold, and how to handle each situation, and never names a tool (a test enforces it). The model decides which tool to call from each tool's own description. A tool is one declaration in `agent/tools.py`: description, schema, handler, plus `requires_verification`, `filler` and `ends_call`. One gate refuses every verification-gated tool, including tools added later. `check_payment_status` was added this way with no prompt or bot change.
7. **The goodbye is spoken by code.** `end_call` has a required `goodbye` field that the bot always speaks. Sonnet 5.5 sometimes returns the text it writes just before a tool call as a hidden reasoning block, which made calls end in silence. Now every call ends with a spoken goodbye (38/38 in the evals).

## Lenders, languages and fixed phrases

Everything the agent says **verbatim** is data, not code. The LLM writes the conversation itself.

```
config/lenders/<lender>/lender.yaml   name, agent name (per script), supported languages
config/lenders/<lender>/<lang>.yaml   greeting, post-verification reminder (due / overdue),
                                      fillers, idle prompts, recovery lines, fallback goodbye,
                                      loan-type names
```

- **New lender:** copy `config/lenders/lumen`, edit the names and wording, and set `LENDER=<folder>` in `.env`. No code change (covered by `tests/test_phrases.py`).
- **New language:** add `<lang>.yaml` and list it in `lender.yaml`, plus a verbaliser in `agent/speech.py` (`VERBALISERS`) that says amounts and dates in that language. Hindi is implemented: 38200 → "अड़तीस हज़ार दो सौ रुपये".
- **Checked at start-up:** each pack is validated when loaded (`agent/phrases.py`): required phrases present, only allowed placeholders (`{first_name}`, `{amount}`, …), and every listed language has a file. A typo fails when the bot starts, not mid-call. A phrase missing for one language falls back to the lender's default language.
- **Why templates at all:** the greeting and the post-verification reminder are the same on every call, so speaking them from the pack skips about 1–1.5s of LLM wait each. Anything that needs judgement (previous calls, failed verification) is still written by the LLM. In production, packs could be drafted by an LLM offline and approved by compliance once, which suits lending, where wording is regulated.

## Memory

| Scope | Store | Contents |
|---|---|---|
| Within a call: what was said | LLM context (in memory) | Full transcript, tool calls and results. Resent to the LLM every turn; calls are short, so no summarising is needed |
| Within a call: facts the guardrails depend on | `CallState` (in memory) | Verification status and attempts, link sent, language, turns, outcome. Owned by code; the LLM can't set it directly |
| Across calls | SQLite, `outcomes/history.db` | One row per call: outcome, promise-to-pay date, callback time, link sent, escalation, notes |

How it flows:
- **Write:** once per call, on `end_call`, idle hang-up or disconnect. `call_id` is the primary key, so a retried save is a no-op. Writes run off the event loop.
- **Read:** inside `verify_identity`, only on success. The last three calls are returned in spoken-ready form (for example, `promised_to_pay_on: "the tenth of November"`, `promise_date_has_passed: true`). The prompt tells the agent to raise a broken promise gently and never as pressure.
- **Evals:** each run gets its own temporary database seeded from the scenario's `history:`, so runs never share or pollute memory.

**Production design (not built, since the assignment isn't an infrastructure exercise):**
- **Redis for live sessions:** `CallState` plus the conversation context, keyed by `call_id` with a time-to-live and checkpointed after every turn. If a worker crashes or the caller reconnects, another worker loads the session. Verified callers skip re-verification; unverified callers restart at identity.
- **Postgres for durable records:** `call_history`, promises to pay and the escalation queue, keeping the same idempotent insert on `call_id`.
- **Retention:** keep transcripts briefly and outcomes for longer; never put loan data in the prompt before verification.

## Tracing (Langfuse)

Optional, off unless Langfuse keys are set. Every call becomes one trace, with no code path depending on it:

```bash
docker compose -f docker-compose.langfuse.yml up -d   # local Langfuse at http://localhost:3000 (admin@lumen.local / lumen-local-admin)
# .env: LANGFUSE_PUBLIC_KEY=pk-lf-local  LANGFUSE_SECRET_KEY=sk-lf-local  (pre-created by the compose file)
```

- **Shape:** Pipecat's OpenTelemetry spans (`conversation` → `turn` → STT / LLM / TTS, with time to first byte and token usage) plus a `tool.<name>` span per tool call under its turn, with redacted input and output (ERROR level when the tool failed).
- **Tags and metadata:** the session id is the call id (the same id as the transcript, recording and outcome record). Tags are `prompt:<version>`, `model:`, `lender:` and `lang:`. `user.id` is a keyed hash of the borrower id, so one borrower's calls group together without the id being in Langfuse. The prompt version is a content hash of the system prompt and every tool description (`agent/prompts.py: prompt_version`), so a regression can be tied to the exact wording that caused it. The outcome, `verified` and `escalate_to_human` are set as trace output and metadata when the call ends.
- **PII:** every span is copied through `RedactingExporter` before it is serialised. DOBs, phone digits, four-digit groups and borrower names are masked in every attribute, transcripts and LLM messages included. `TRACE_CONTENT=none` drops conversation text entirely and keeps only timings, tags and outcomes. Names written in Devanagari aren't caught yet.
- **Never breaks a call:** spans are exported on a background thread with a 5s timeout. An unreachable Langfuse only loses spans (a test proves export returns within the timeout, and a broken exporter is contained). Every helper swallows its own errors.

## Consent and data handling

**Consent to record.** When recording is on (`RECORD_CALLS=true`, the default), the greeting says so before anything is asked: "Hello, this is Asha calling from Lumen Finance. This call is recorded for quality and training. Am I speaking with Rahul Sharma?" (the Hindi pack has its own line). The wording lives in the phrase pack (`recording_notice`), so compliance can change it without a code change. Gaps: the notice is information, not an opt-in. A caller who objects isn't handled automatically: the recording can't yet be stopped mid-call, and the objection should be flagged for the team. With templated turns off, the LLM writes the greeting and isn't told to give the notice. Set `RECORD_CALLS=false` to record nothing.

**What is stored, where, and for how long**

| Data | Where | Contains | Kept |
|---|---|---|---|
| Call audio | `outcomes/recordings/<call_id>.wav` | Both voices | `RETENTION_DAYS` (30), then deleted |
| Transcript | `outcomes/transcripts/<call_id>.{txt,json}` | Everything said, including DOB or phone digits, plus tool results (loan details) | `RETENTION_DAYS` (30) |
| Outcome record | `outcomes/history.db` (`call_history`) | Outcome, promise date, notes, prompt version; no transcript or audio | Until erased: cross-call memory needs it |
| Identity checks | `history.db` (`verification_attempts`) | Pass/fail per check, never the values given | `RETENTION_DAYS`; only the last 24h matter for lockout |
| Logs | `outcomes/bot.log` | Redacted since 21 Nov (DOBs, digits, names masked) | Rotated at 20 MB; rotated files deleted after `RETENTION_DAYS` |
| Latency, usage, turns | `outcomes/*.jsonl` | Timings and token counts only | Not purged (no personal data) |
| Traces (optional) | Langfuse | Redacted spans; borrower id hashed; `TRACE_CONTENT=none` keeps no text | Langfuse's own retention settings |
| Audit log | `outcomes/audit.jsonl` | Who, what, which borrower id, when; no PII | Kept (it's the record of deletions too) |

- **Retention.** `uv run python -m agent.retention --dry-run` lists what is past `RETENTION_DAYS`; without `--dry-run` it deletes it. The bot also runs it in the background at the start of every call, so nothing piles up. `RETENTION_DAYS=0` keeps everything.
- **Erasure (a borrower's deletion request).** `uv run python -m agent.retention erase LF-1001` removes their call history, identity checks, transcripts and recordings, finding calls through both the database and the transcripts. The audit entry is written first: if it can't be written, nothing is deleted.
- **Audit log.** Every read or change of borrower data is recorded:
  - the borrower page (each person's own token via `ADMIN_TOKENS=alice:…,bob:…`, so entries name who);
  - the CLI tools that show DOBs, history or transcripts (`cli:<os user>`);
  - the agent itself whenever it releases loan details to a verified caller, or locks out a caller after failed checks;
  - online evals reading transcripts;
  - every retention or erasure deletion.

  Entries are hash-chained: `uv run python -m agent.audit --verify` detects an edited, removed or reordered entry. `uv run python -m agent.audit --borrower LF-1001` shows everything that touched one borrower.
- **Who processes call data.** Deepgram (caller audio), Anthropic (the conversation text, and the simulator and judge in evals), and Cartesia (Asha's lines, which contain loan details after verification). A real deployment needs data-processing agreements with each, and Indian data residency where it applies.

## Evals

Full write-up: [REPORT.md](REPORT.md).

- `evals/scenarios.yaml`: **20 scenarios**, each with a persona, behaviour and expected outcome record. They cover every response the brief lists (pay on time, already paid, promise to pay, hardship, dispute, wrong number, family member, callback, rude/off-topic, Hindi switch) and the hard cases: failed and misheard verification, stop calling, prompt injection, a terse hardship caller, a garbled date, three memory scenarios (broken promise, history not leaked to family, previous hardship now paying), and a payment-link failure.
- `evals/simulator.py`: an LLM (Claude Opus 5.5) plays the borrower, in text.
- `evals/text_agent.py`: the same prompt, tools, guardrails, templates and goodbye handling as the live bot, minus audio. Fault injection (`inject_failures:` in a scenario) makes a tool fail through the real error path.
- `evals/metrics.py`: rule checks on the outcome record and transcript, plus an LLM judge.

| Metric | Method | Trust |
|---|---|---|
| Outcome / verified / escalation / promise-to-pay date / link sent correct | Compare the saved outcome record with the scenario's expected record | High |
| No disclosure before verification | Scan every agent line said before verification for the loan amount, due date, loan type and past-call details, in digits and in words, English and Hindi | High (may miss paraphrases) |
| Required mentions (e.g. raising a broken promise) | Pattern match with alternatives | Medium |
| No meta-narration, raw digits, polite close, single `end_call` | Rule checks on the transcript and tool calls | High |
| Tone, empathy, naturalness, guardrail violations | LLM judge with a structured rubric | Medium; spot-checked by hand. `evals/calibrate.py` measures it against human labels (Cohen's kappa) |
| Latency (live calls) | `UserBotLatencyObserver` per turn, `evals/latency_report.py` | High |

### Online evals: scoring real calls

`uv run python -m evals.online` scores the day's real calls (IST) from their saved transcripts (`--all` for every call, `--no-judge` for the free rule checks only). Rule checks: no loan details before verification (past-call dates included), a goodbye was spoken (not required when the caller hung up), the outcome record is consistent with what happened (a promise has a date, a "sent" link was actually sent, hardship and disputes escalated, the agent closed with `end_call`), and no banned phrases (legal action, police, visiting the home, telling family). The judge adds tone, empathy, naturalness, guardrail violations and whether the recorded outcome matches what the borrower said. Scores go to Langfuse on the call's session, with stable ids so a re-run updates them. `results/online/<date>.md` lists each call flagged for human review, with the reason, the evidence line and the transcript path.

First run, over all 39 saved live calls: 13 flagged. 11 were missing goodbyes: 10 of 23 calls on 16 Nov, before the `goodbye` field fix, and 1 of 9 since, which ended on the filler "Sure, I'm sending that to you now" instead of a closing. One was an early call whose outcome record was saved incomplete. One was Sneha's hardship call (empathy 1: "hung up with a curt 'Thank you for your time'"). The judge found the recorded outcome matched the conversation on 39/39 calls, with 0 guardrail violations and 0 disclosures before verification. It also noticed the agent switching from Hindi to English mid-call. Writing these checks surfaced one false positive ("have a good day" wasn't counted as a goodbye), which is now fixed and tested.

### Judge calibration

`uv run python -m evals.calibrate template results/online/all.jsonl --sample 30` writes `results/calibration/labels.csv`: one row per call, with the judge's verdicts left out so the reviewer isn't anchored by them. A person fills in tone, empathy and naturalness (1–5) and guardrail_violation and outcome_correct (yes/no). Then `uv run python -m evals.calibrate compare results/calibration/labels.csv results/online/all.jsonl` reports per criterion: raw agreement, Cohen's kappa, and for the 1–5 scores agreement within one point plus quadratic-weighted kappa. For yes/no criteria it counts each direction of disagreement ("judge missed a violation" is worse than "judge flagged a clean call"). It works on offline eval runs too (`results/<run>/runs.jsonl`). **Human labels haven't been collected yet**, so the judge's trust level above is still "medium, spot-checked". One data point so far: two judge runs over the same 39 calls gave empathy means of 3.46 and 3.54.

### CI and prompt versions

`.github/workflows/ci.yml` runs on every pull request and every push to `main`:
- **Unit tests** (`uv run pytest`).
- **An eval gate:** the 7 scenarios tagged `subsets: [ci]` in `scenarios.yaml` (pay on time, promise to pay, hardship, family member, failed verification, prompt injection, payment-link failure), 1 run each, rule checks only (about $0.30). `--min-pass-rate 0.85` fails the build if fewer than 6 of 7 pass. It needs an `ANTHROPIC_API_KEY` repository secret and is skipped without one (for example, on pull requests from forks). Results are uploaded as a build artifact.

Every outcome record, history row, eval summary and trace carries `prompt_version`. It is the `PROMPT_VERSION` constant in `agent/prompts.py`, pinned in `agent/prompt_versions.json` to a hash of the system prompt and every tool description. Editing any of that wording without bumping the version fails a unit test, so a prompt change is always deliberate, and a regression can be traced to the wording that caused it. A run with un-pinned edits is labelled `goals-v1+dirty-<hash>` rather than passing as the old version.

### STT accuracy (WER) on amounts, dates and digits

`uv run python -m evals.wer` (`--engine nova` for the fallback STT) plays 21 sentences callers actually say, in English, Hindi and Hinglish (`evals/wer_cases.yaml`: DOBs, last-four digits, amounts, payment dates), through the live bot's Flux settings in real time. It reports WER and **entity accuracy**: whether each date, digit string and amount came through, in any spelling (digits, English words or Hindi words).

| STT | Language | WER raw / normalised | Entities correct |
|---|---|---|---|
| Flux multilingual (live) | English | 1% / 1% | 13/13 |
| | Hindi | 9% / 0% | 10/10 |
| | Hinglish | 2% / 2% | 5/5 |
| Nova-3 multi (fallback) | English / Hindi / Hinglish | 2% / 13% / 0% raw | 28/28 |

"Normalised" maps month names and number words to one token in either script: Flux often writes "December" in a Hindi sentence, which is correct. **These are upper bounds.** The clips are synthesised by Cartesia, which is much cleaner than a caller on a phone line, and live transcripts do show garbling ("pay on the pixel", handled by the confirm-back rule). Each case can point to a real recording (`audio:`). Recording 20 real clips, ideally over a phone line, is the way to get real numbers.

**Latest full run (Sonnet 5.5):** 38/38 runs pass, 0 guardrail violations, tone 4.87, empathy 3.97, naturalness 4.32; every goodbye spoken; prompt cache hit on 153 of 158 LLM calls. Payment-link failure: 3/3, tone 5.0.

## Latency work and findings

### Current latency, layer by layer

Measured on live browser calls from India with the current setup (Flux, Sonnet 5.5, Cartesia, templated turns): 25 turns across 8 calls. Each turn runs from the moment the caller stops speaking until Asha's first audio plays. Source: `UserBotLatencyObserver`, logged per turn to `outcomes/latency.jsonl`.

| Layer | What happens | p50 | p95 | Range |
|---|---|---|---|---|
| Endpointing wait | VAD confirms the caller has gone quiet (`stop_secs`) | 0.20s | 0.20s | fixed |
| Speech-to-text | Deepgram Flux final transcript (includes ~0.35s network round trip to the US) | 0.27s | 0.65s | 0.10–0.92s |
| Turn detection | Flux decides the turn is over (eager end-of-turn; the LLM already started) | 0.00s | 0.01s | |
| LLM | Claude Sonnet 5.5, time to first spoken token (cached prompt, low effort) | 1.09s | 2.14s | 0.71–5.98s (the max was one uncached first turn, now fixed) |
| LLM tool call (tool turns only) | Generating the tool call before the filler or template plays | 0.57s | 2.84s | 0.23–2.84s (9 turns) |
| Tool handler | Our code (`verify_identity`, `check_payment_status`) | 0.01s | 0.01s | |
| Sentence wait (sentence mode only) | First full sentence before TTS starts | 0.68s | | one turn measured |
| Text-to-speech | Cartesia Sonic-3, time to first audio | 0.29s | 0.74s | 0.12–0.74s |
| Pipeline | Frame handling, transport | 0.13s | 0.63s | 0–0.64s |
| **Total, voice to voice** | | **2.25s** | **4.72s** | 1.42–7.17s; p90 2.87s |

Percentiles are nearest-rank, from `uv run python -m evals.latency_report`. Split by turn type: replies the LLM answers directly have a p50 of **2.22s**; turns with a tool call have a p50 of **2.37s**. On tool turns the caller hears a filler line ("Thanks, one moment") at that point, so the wait for the real answer is covered. With 25 turns, p95 is set by the two slowest, so it will move as more calls are logged.

**Interruption rate** (the share of Asha's turns the caller cut off) is logged per turn to `outcomes/turns.jsonl` from 21 Nov and shown by the same report. The calls above predate it, so there is no number yet.

The LLM is about half of every turn. The model comparison in finding 14 showed that the fastest open-weight models would save about 0.3s, but they failed the safety checks. The next largest fixed cost is the network round trip from India to US-hosted STT and LLM endpoints.

### How we got here

The first live call measured **2.6–5.1s** from the caller finishing to the agent starting to speak. Breakdown:
- LLM time to first token (Opus 5.5): 1.1–2.0s
- One Smart Turn stall: 2.5s
- Waiting for a full sentence before TTS: 0.5–0.7s
- Deepgram final transcript: about 0.45s
- Cartesia first audio: about 0.12s

Changes:
- **Model.** Measured with `uv run python -m evals.bench_llm` (time to first spoken token, warm cache, median of 5):

  | Setting | Greeting | Normal reply | Reply after a tool |
  |---|---|---|---|
  | Opus 5.5, low effort | 0.86s | 1.36s | 1.68s |
  | **Sonnet 5.5, low effort, adaptive (chosen)** | **0.76s** | **0.78s** | **1.14s** |
  | Sonnet 5.5, reasoning off (`between_tools`) | 1.58s | 1.81s | 1.80s |

  Surprise: turning reasoning off was *slower* on every turn type, so we keep adaptive reasoning at low effort.
- **Prompt caching and pre-warm.** The pre-warm writes about 2.3k tokens of tools and system prompt to the cache before the greeting. In the evals, every LLM call after the first in a conversation reads from the cache.
- **Token streaming into TTS (later reverted, see finding 13).** This removes the 0.5–0.7s sentence wait. The trade-off is that the speech formatter only sees fragments, so numbers must arrive already in words: `verify_identity` returns spoken forms and the prompt asks for words. The eval counts agent lines with raw digits or ₹ symbols; it has been 0 so far.
- **STT/TTS warm start.** Deepgram and Cartesia already open their websockets when the call starts, before the greeting. Preloading VAD and Smart Turn takes 0.06s, so that was never on the critical path.

Failures the evals found:
1. **A caller who says "busy, bye" lost their promise date.** Asha asked one more question, the caller hung up, `end_call` never ran, and the 30 November promise was not recorded. Fix: a prompt rule to confirm the plan and close in the same turn once the caller wants to leave. The eval now treats a hang-up exactly like the live bot does, so the problem shows up in the scores.
2. **Too-eager Hindi.** A single "haan" made Asha switch an English-preferring borrower to full Hindi. Fix: switch only on request or on whole sentences in another language. After the fix, English callers got 0 Hindi lines and the Hindi-switch scenario still switches.
3. **A network blip left the caller in 17 seconds of silence (live call).** The Deepgram and Cartesia websockets broke together. Deepgram reconnected after about 5s, but the reply being synthesised was silently dropped (Cartesia's time-to-first-byte showed 16s: the timer only stopped when the caller said "Hello?"). Fix: `agent/recovery.py`. The first version retried after a fixed 1s and failed in a live network-switch test: Cartesia took about 10s to notice its socket was dead, so the retry was dropped too, and a cooldown then suppressed the STT apology. The current version waits for the reconnect event and confirms the audio actually played. `tests/test_recovery.py` replays both live incidents.
4. **Memory derailed a call (live call).** Rahul had a hardship call in his history from earlier that day. Right after the reminder, Asha switched to asking whether the specialist had called, never asked about this EMI, and recorded `hardship` again: a copy of the earlier outcome, not this call's. Fix: previous calls are background, not the agenda; the reminder and intent come first, and the outcome describes this call only. New eval `prior_hardship_now_paying` passes 3/3, and the broken-promise memory scenarios still pass.
5. **Silent and abrupt endings: an LLM API behaviour, found by looking at the response blocks.** On short calls and after tool calls, Asha hung up without a goodbye. Worst case (live): Sneha said "I lost my job, I really can't pay" and Asha hung up immediately. Prompt fixes only partly helped (wrong-number calls went from 5/8 silent to 0/12, but closings after the payment link still went silent in about 4 of 9 runs). Logging the content blocks showed the cause: on Sonnet 5.5, text written right before a tool call can come back as a hidden progress-update thinking block, so the reply after `send_payment_link` was `[thinking, tool_use(end_call)]` with no text, and the spoken "link sent, thank you, goodbye" never reached TTS. Fix: `end_call` now carries a required `goodbye` field that the bot speaks itself before hanging up, unless the agent already said goodbye as text. Result: 18/18 regression runs pass, every call closes with a confirmation and goodbye, judge tone for disputes rose from 3.0 to 4.0, and a new `hardship_terse` eval (a borrower who says one sentence and goes quiet) passes.
6. **Small live fixes.** One- or two-word replies ("Payment") no longer flip the language. Verification is only attempted with a complete date of birth (an incomplete "fourteen ninety" used to burn an attempt). Tool results no longer appear as caller lines in transcripts.
7. **Haiku 4.5 vs Sonnet 5.5: faster but unsafe.** Time to first token (interleaved, 8 runs per turn): Haiku 0.68–0.84s vs Sonnet 1.08–1.57s (2.64s on a turn where Sonnet reasons first). Haiku gets no prompt caching (our ~2.4k-token prompt is below its 4,096-token minimum), and is still faster. Full eval suite (18 scenarios × 2 runs): Sonnet 36/36 with 0 judge-flagged violations, Haiku 32/36 with 7. Haiku asked a caller who said "please stop calling" to verify first (2/2 runs; she hung up and nothing was recorded), mentioned the loan before verification in Hindi with the wrong gender ("मैं समझ गया, भाई…"), narrated its tool use aloud ("Let me verify that for you…"), and labelled a family member as `wrong_number`. Decision: stay on Sonnet; the ~0.5s gain is not worth guardrail failures.
8. **Reasoning off (`between_tools`) vs adaptive.** Live, `between_tools` removed the reasoning spikes (12 turns: p90 2.29s vs 3.19s), but the agent **read out its own notes**: "The caller corrected themselves and gave a complete date of birth." Even after adding a "never narrate" prompt rule, a full eval (18 scenarios × 2 runs per mode, same prompt and judge) showed adaptive at 35/36 with 0 narration, against between_tools at 31/36 with 2 narrated notes ("Caller wants to go, so I'll confirm and close in this turn"), 4 closings without a goodbye, and one rude-caller run where the agent asked another question after "bye". Decision: adaptive at low effort. A side bug found here: when the model spoke its own closing, the code also spoke the `goodbye` field, and the two interleaved word by word. The field is now only spoken when the model said nothing.
9. **Noise A/B test (live, background noise playing, same script).** RNNoise on (3 calls): caller-channel noise floor −90 dBFS when nobody spoke, no noise reaching the bot while Asha spoke, 0 false interruptions. RNNoise off (1 call): noise floor −43 dBFS (about 17 dB below the voice), 2.6s of noise while Asha spoke. Mishearings happened in both ("Sporting March", "pay on the pixel" / "thirteen March", "pay on sir"), so they come from the noise, not the filter. Decision: RNNoise on. The test also showed Asha confirming "the pixel" in one call but silently accepting "pay on sir" in another. A new rule (read back any date, amount or digits that look garbled) and the `garbled_date` eval (3/3, "Did you say you'll pay on the fifth of December?") fix that.
10. **Templated turns and a TTS buffer fix (latency options A and B).** The greeting and the post-verification reminder (English, no call history) are now spoken from templates (`agent/templates.py`) without waiting for the LLM. The reminder used to need a second LLM pass after `verify_identity`. The LLM still writes every turn that needs judgement (history, Hindi, failed verification). Cartesia's server buffer, left at its 3000ms default in token-streaming mode, is now a 150ms window (`CARTESIA_MAX_BUFFER_MS`). Full eval: 38/38 passing, 0 violations, judge tone 4.56 → 4.84, naturalness 4.25 → 4.42, about one fewer LLM wait per call.
11. **Scripted prompt → goals-based prompt plus a tool registry.** The prompt used to name each tool and walk through a step-by-step script, so adding a tool meant editing the prompt. Now the prompt states the call's goals, the rules that always hold, and how to handle each situation, and **never names a tool** (a test enforces this). Each tool is one declaration in `agent/tools.py`: when to use it (its description, which the model reads), its schema and handler, plus `requires_verification`, `filler` and `ends_call`. One generic gate refuses every verification-gated tool, including future ones. Proof: a new tool, `check_payment_status`, was added with no prompt or bot change; the model used it only in the "already paid" and "dispute" scenarios. The first full eval after the rewrite dropped (36/38, tone 4.66): "already paid" fell to 3.0 because the model checked the payment and then hung up. The response blocks showed the same Sonnet 5.5 issue as before: its answer was a hidden reasoning block next to `end_call`. Fix, in the tool's description only: `goodbye` must carry everything still to say, including the answer to the caller's last question. Final full eval: **38/38, 0 violations, tone 4.87, empathy 3.97 (best so far), naturalness 4.32**.
12. **An eval bug, not an agent bug.** The broken-promise phrase check wanted "tenth of November" and missed "the tenth" and the Hindi "दस तारीख". The check now accepts alternatives.
13. **The first reply after the greeting started with a stretched "Thank…" (live, measured in a phone recording).** Two causes. (a) `end_call` was registered as asynchronous, so Pipecat appended an "ASYNC TOOLS" section to the system prompt at pipeline start, after the cache pre-warm. The first LLM turn of every call missed the cache (2.15s to first token instead of about 1.1s). All tools are now synchronous, and a check confirms the composed prompt is identical to the pre-warmed one. (b) In token streaming mode, Cartesia started speaking on the lone first token: in the recording, "Thank" lasted 0.88s, split from "you, Ananya". TTS now gets whole sentences. This costs about 0.3s on LLM-written turns only (0.68s waiting for the sentence, minus about 0.35s faster Cartesia start), and speech sounds natural.
14. **Open-weight models: faster, but not safe enough for this call.** Same 19 scenarios × 2, same judge, via `--agent-model <provider>:<model>`. Time to first spoken token is a streaming benchmark on the same prompt from India.

    | Model | Runs passing | Guardrail violations | Tone / empathy / naturalness | Time to first token |
    |---|---|---|---|---|
    | **Claude Sonnet 5.5** | **38/38** | **0** | 4.87 / 3.97 / 4.32 | 0.89s |
    | Qwen 3.8 27B (Cerebras) | 31/38 | 1 | 4.63 / 3.66 / 4.11 | 0.64s |
    | gpt-oss-120b (Cerebras) | 14/38 | 6 | 3.71 / 2.76 / 2.42 | 0.56s |
    | gpt-oss-120b (Groq, 6 scenarios) | 1/6 | — | — | (free tier, rate limited) |

    Qwen told a borrower's wife that Amit has a loan. gpt-oss invented the borrower's answer and verified with it ("5675"), read raw JSON aloud, and invented a complaint reference number. On Groq it crashed on malformed tool calls (null for a string field, unparseable JSON) in 4 of 6 calls. Both models score about 98% on Pipecat's general voice benchmark; our domain checks (privacy, outcome recording) separate them from Sonnet. A saving of about 0.3s per reply is not worth disclosing a debt to a family member, so Sonnet stays.

## Cost per call

Measured from 5 complete live calls (about 1 minute each), using the token, character and audio-second counts Pipecat logs per call, multiplied by list prices (November 2026).

| Layer | Service | Unit price | Usage per call | Cost per call | Share |
|---|---|---|---|---|---|
| Speech-to-text | Deepgram Flux multilingual | $0.0078 / min of caller audio | ~1 min streamed | $0.008 | 16% |
| LLM | Claude Sonnet 5.5 | $2 / $10 per M input / output tokens; cache reads 10% of input, cache writes 125% | 4–5 requests, ~17k cached tokens read, ~1–4k written, ~400 output | $0.014 | 29% |
| Text-to-speech | Cartesia Sonic-3 | ~$0.04 / 1k characters (Startup plan; Pro ~$0.05) | ~680 characters spoken | $0.027 | 55% |
| VAD, turn model, noise filter | Silero, Smart Turn v3, RNNoise | Free (open source, run locally on CPU) | | $0 | |
| Transport | Pipecat SmallWebRTC (browser) | Free | | $0 | |
| **Total** | | | | **≈ $0.05 per call (≈ $0.05 per minute)** | |

That is about **$49 per 1,000 calls**, or about $4,900 for 100k calls a month. A production phone deployment would add telephony (typically about $0.01 per minute) and hosting.

What drives it:
- **TTS is the largest cost**, not the LLM. Shorter agent turns cut cost and latency together. Templated lines can't be cached by Cartesia today, but they are a candidate for pre-rendered audio (greeting, fillers, reminders), which would remove roughly a third of TTS characters.
- **Prompt caching keeps the LLM cheap.** About 95% of input tokens are cache reads; without caching the LLM cost would be roughly 4× higher.
- The Sonnet rates above are the assumed list price for this model class; check the Anthropic console. At $3 / $15 the LLM rises to about $0.02 per call, and the total to about $0.056.
- **Switching to a cheaper open-weight LLM saves at most about 29% of the bill**, so model choice here is a latency and safety decision, not a cost one (see finding 14 above).

## Project layout

```
agent/      bot.py (Pipecat pipeline), prompts.py (goals-based prompt), tools.py (tool registry),
            templates.py, phrases.py, speech.py, call_state.py, history.py, recovery.py,
            redact.py (PII masking), compliance.py (calling hours), tracing.py (Langfuse over OTel),
            audit.py (hash-chained audit log), retention.py (retention and erasure),
            recording.py, transcripts.py, usage_observer.py, borrowers.py, config.py,
            web.py (borrower picker), use_borrower.py
config/     lenders/<lender>/{lender,en,hi}.yaml: names and every fixed phrase, per language
data/       borrowers.json (8 mock borrowers), payments.json (mock payment records),
            call_history_seed.json (demo history)
evals/      scenarios.yaml, simulator.py, text_agent.py (Claude + OpenAI-compatible hosts),
            metrics.py, run_evals.py, latency_report.py, bench_llm.py,
            online.py (scores real calls), calibrate.py (judge vs human labels),
            wer.py + wer_cases.yaml (STT accuracy)
tests/      unit tests: guardrails, registry, phrase packs, templates, call history, recovery,
            recording, noise filter, security (lockout, dates, PII, admin page, calling hours), tracing
docker-compose.langfuse.yml   local Langfuse for traces
.github/workflows/ci.yml      unit tests + eval gate on every PR
agent/prompt_versions.json    PROMPT_VERSION -> prompt hash, with a note per version
```
