"""Runtime configuration, read once from the environment."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # LLM: the agent model is the single biggest latency lever (time to first token),
    # so it is configurable and benchmarked rather than hard-coded.
    agent_model: str = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")
    agent_effort: str = os.getenv("AGENT_EFFORT", "low")
    # "adaptive": the model may think briefly on hard turns (skips it on most simple ones at low effort).
    # "between_tools": no extended thinking at all (Sonnet 5.5 only), lowest time to first token.
    agent_thinking: str = os.getenv("AGENT_THINKING", "adaptive")
    # Prompt caching for the system prompt + tools + conversation so far.
    prompt_caching: bool = os.getenv("PROMPT_CACHING", "true").lower() == "true"
    simulator_model: str = os.getenv("SIMULATOR_MODEL", "claude-opus-5-5")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-opus-5-5")

    # "flux": Deepgram Flux does transcription AND turn detection, and predicts the end of a
    #         turn early so the LLM starts speculatively (discarded if the caller keeps talking).
    # "nova": Nova-3 transcription + local Smart Turn v3 for turn detection (no speculation).
    stt_engine: str = os.getenv("STT_ENGINE", "flux")
    flux_model: str = os.getenv("FLUX_MODEL", "flux-general-multi")  # multilingual incl. Hindi
    flux_eager_eot_threshold: float = float(os.getenv("FLUX_EAGER_EOT_THRESHOLD", "0.5"))
    flux_eot_threshold: float = float(os.getenv("FLUX_EOT_THRESHOLD", "0.7"))
    deepgram_model: str = os.getenv("DEEPGRAM_MODEL", "nova-3")
    deepgram_language: str = os.getenv("DEEPGRAM_LANGUAGE", "multi")
    cartesia_voice_id: str = os.getenv("CARTESIA_VOICE_ID", "")
    # "sentence": send whole sentences to TTS (natural prosody; first sentences are short, so the
    #             wait is small). "token": stream tokens (starts sooner, but a lone first token
    #             like "Thank" gets stretched by the voice).
    tts_text_mode: str = os.getenv("TTS_TEXT_MODE", "sentence")
    # Cartesia's server-side text buffer before it starts speaking. In token mode Pipecat leaves
    # it unset, which means Cartesia's 3000ms managed buffer; a short window starts audio sooner.
    cartesia_max_buffer_ms: int = int(os.getenv("CARTESIA_MAX_BUFFER_MS", "400"))
    # Speak the greeting and the post-verification reminder from templates (no LLM wait).
    templated_turns: bool = os.getenv("TEMPLATED_TURNS", "true").lower() == "true"

    # Conversation timing
    # Smart Turn decides "finished vs pausing" from audio. When it judges a turn incomplete, it
    # waits this long in silence before ending the turn anyway. Pipecat's default is 3.0s; live
    # calls showed 11 of 57 turns stalled ~2.7s on short, complete answers ("Hello?", "Yes.").
    # Lower = snappier, but a longer mid-sentence pause may be cut off.
    smart_turn_stop_secs: float = float(os.getenv("SMART_TURN_STOP_SECS", "1.2"))
    user_idle_timeout_secs: float = float(os.getenv("USER_IDLE_TIMEOUT_SECS", "7"))
    max_idle_prompts: int = int(os.getenv("MAX_IDLE_PROMPTS", "2"))

    # Mock tool behaviour (lets us demo and eval dead-air handling and tool failures)
    payment_link_delay_secs: float = float(os.getenv("PAYMENT_LINK_DELAY_SECS", "2.5"))
    payment_link_failure_rate: float = float(os.getenv("PAYMENT_LINK_FAILURE_RATE", "0"))
    payment_check_delay_secs: float = float(os.getenv("PAYMENT_CHECK_DELAY_SECS", "1.0"))

    # Borrower to call when the browser client doesn't pick one
    default_borrower_id: str = os.getenv("DEFAULT_BORROWER_ID", "LF-1001")

    record_calls: bool = os.getenv("RECORD_CALLS", "true").lower() == "true"
    # Which lender's phrase packs to use: config/lenders/<LENDER>/
    lender: str = os.getenv("LENDER", "lumen")
    # Noise suppression on the caller's audio before VAD and STT see it.
    # "rnnoise": open-source RNN denoiser, runs locally on CPU. "off": raw mic audio.
    noise_filter: str = os.getenv("NOISE_FILTER", "rnnoise")

    borrowers_path: Path = ROOT / "data" / "borrowers.json"
    outcomes_dir: Path = ROOT / "outcomes"
    history_db: Path = ROOT / os.getenv("HISTORY_DB", "outcomes/history.db")  # absolute paths win


CONFIG = Config()


def model_supports_effort(model: str) -> bool:
    """Haiku 4.5 rejects output_config.effort (and has no adaptive thinking); newer models take it."""
    return not model.startswith("claude-haiku-4-5")
