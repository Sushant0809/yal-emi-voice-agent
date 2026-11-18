"""Recover the conversation when the speech providers drop mid-call.

Seen on live calls: a network blip broke the Deepgram and Cartesia websockets
together. Deepgram reconnected in ~5s, Cartesia only noticed its dead socket and
reconnected after ~10s, and replies synthesised in between were dropped unspoken.

Recovery is driven by real signals, not timers:
- An outage creates one *pending* recovery line: "sorry, the line broke up" plus
  the reply that was lost (TTS drop) or a request to repeat (STT drop).
- The line is spoken when the TTS reconnects (its on_connected event). For an
  STT-only outage the TTS is fine, so it is spoken after a short delay.
- It counts as recovered only when its audio actually starts playing. If it is
  dropped again, it stays pending for the next reconnect (bounded attempts).
- If the caller speaks first, or the agent speaks normally, the pending line is
  discarded: the conversation has moved on and the LLM answers naturally.
"""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

from agent.phrases import lender

MAX_ATTEMPTS = 3

_CONTEXT_ID = re.compile(r"TTS context (\S+) completed with no audio")


def context_id_from_error(error: str) -> str | None:
    """Pipecat reports a dropped reply as 'TTS context <id> completed with no audio'."""
    match = _CONTEXT_ID.search(error)
    return match.group(1) if match else None


class CallRecovery:
    def __init__(
        self,
        *,
        call_id: str,
        speak: Callable[[str], Awaitable[None]],
        language: Callable[[], str],
        log_path: Path,
        on_incident: Callable[[], None] = lambda: None,
        stt_only_delay: float = 1.5,
        reconnect_settle: float = 0.3,
    ):
        self._call_id = call_id
        self._speak = speak
        self._language = language
        self._log_path = log_path
        self._on_incident = on_incident
        self._stt_only_delay = stt_only_delay
        self._reconnect_settle = reconnect_settle

        self._texts: dict[str, list[str]] = {}  # TTS context id -> text chunks sent for it
        self._pending: str | None = None  # the line owed to the caller
        self._recovery_lines: set[str] = set()  # lines we spoke ourselves; never re-wrapped
        self._attempts = 0
        self._in_flight = False  # pending line queued, waiting to hear its audio start
        self._tts_down = False
        self._tasks: set[asyncio.Task] = set()
        self.incidents = 0

    # --- inputs from the pipeline ------------------------------------------------

    def on_tts_request(self, context_id: str, text: str) -> None:
        self._texts.setdefault(context_id, []).append(text)
        if len(self._texts) > 50:  # only recent contexts matter
            self._texts.pop(next(iter(self._texts)))

    def on_tts_no_audio(self, error: str, context_id: str | None) -> None:
        text = "".join(self._texts.pop(context_id, [])).strip() if context_id else ""
        self._tts_down = True
        self._log("tts_no_audio", error)
        if self._in_flight:
            # Our own recovery line was dropped too: keep it pending for the next reconnect.
            self._in_flight = False
            self._log("recovery_line_dropped", "will retry when the TTS reconnects")
            return
        if text and text not in self._recovery_lines:
            self._set_pending(f"{self._t('recovery.apology')} {text}")

    def on_stt_error(self, error: str) -> None:
        self._log("stt_error", error)
        self._set_pending(self._pending or self._t("recovery.ask_again"))
        if not self._tts_down:
            # STT-only outage: the voice still works, so ask right away (after a beat).
            self._spawn(self._deliver(self._stt_only_delay))

    def on_tts_connected(self) -> None:
        if not self._tts_down:
            return  # initial connect, not a reconnect
        self._tts_down = False
        self._log("tts_reconnected", "delivering pending line" if self._pending else "nothing pending")
        if self._pending:
            self._spawn(self._deliver(self._reconnect_settle))

    def on_user_started_speaking(self) -> None:
        if self._pending and not self._in_flight:
            self._log("recovery_cleared", "caller spoke first; the LLM will answer them")
            self._clear()

    def on_bot_started_speaking(self) -> None:
        if self._in_flight:
            self._log("recovered", self._pending or "")
        elif self._pending:
            self._log("recovery_cleared", "agent spoke normally")
        self._clear()

    # --- recovery actions ----------------------------------------------------------

    def _set_pending(self, line: str) -> None:
        if self._pending is None:  # one incident per outage, however many errors it causes
            self.incidents += 1
            self._on_incident()
        self._pending = line
        self._attempts = 0

    async def _deliver(self, delay: float) -> None:
        await asyncio.sleep(delay)
        if not self._pending or self._in_flight or self._tts_down:
            return
        if self._attempts >= MAX_ATTEMPTS:
            self._log("recovery_gave_up", self._pending)
            self._clear()
            return
        self._attempts += 1
        self._in_flight = True
        self._recovery_lines.add(self._pending)
        self._log("recovery_speaking", self._pending)
        await self._speak(self._pending)

    def _clear(self) -> None:
        self._pending = None
        self._attempts = 0
        self._in_flight = False

    def _t(self, key: str) -> str:
        return lender().phrase(self._language(), key)

    # --- plumbing --------------------------------------------------------------------

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def cancel(self) -> None:
        for task in list(self._tasks):
            task.cancel()

    def _log(self, kind: str, detail: str) -> None:
        logger.warning(f"recovery[{kind}]: {detail}")
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a") as f:
                f.write(json.dumps({"call_id": self._call_id, "t": time.time(), "kind": kind, "detail": detail}) + "\n")
        except OSError:
            pass


class BotSpeechObserver(BaseObserver):
    """Tells CallRecovery when audio actually starts playing to the caller."""

    def __init__(self, recovery: CallRecovery, **kwargs):
        super().__init__(**kwargs)
        self._recovery = recovery
        self._seen: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        if isinstance(data.frame, BotStartedSpeakingFrame) and data.frame.id not in self._seen:
            self._seen.add(data.frame.id)
            self._recovery.on_bot_started_speaking()
