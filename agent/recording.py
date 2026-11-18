"""Save every live call as a WAV file: outcomes/recordings/<call_id>.wav

Stereo: the caller on the left channel, Asha on the right, so the file plays as a
normal two-sided call but either voice can still be isolated for review.
Pipecat's AudioBufferProcessor captures both sides inside the pipeline and flushes
the buffered audio when the call ends (EndFrame or CancelFrame).
"""

import asyncio
import wave
from pathlib import Path

from loguru import logger
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from agent.config import CONFIG

RECORDINGS_DIR = CONFIG.outcomes_dir / "recordings"


class CallRecorder:
    def __init__(self, call_id: str):
        self.call_id = call_id
        self.processor = AudioBufferProcessor(num_channels=2, auto_start_recording=True)
        self._chunks: list[bytes] = []
        self._sample_rate = 0
        self._channels = 2

        @self.processor.event_handler("on_audio_data")
        async def on_audio_data(_processor, audio: bytes, sample_rate: int, num_channels: int):
            self._chunks.append(audio)
            self._sample_rate, self._channels = sample_rate, num_channels

    async def save(self) -> Path | None:
        await self.processor.stop_recording()  # no-op if the call's end already flushed it
        # Pipecat runs on_audio_data in background tasks, and the whole call arrives in one
        # flush at the end; wait for those tasks so the file is never written half-empty.
        pending = [t for name, t in getattr(self.processor, "_event_tasks", set()) if name == "on_audio_data"]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if not self._chunks or not self._sample_rate:
            return None
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = RECORDINGS_DIR / f"{self.call_id}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(self._channels)
            wav.setsampwidth(2)  # 16-bit PCM
            wav.setframerate(self._sample_rate)
            wav.writeframes(b"".join(self._chunks))
        secs = sum(len(c) for c in self._chunks) / (2 * self._channels * self._sample_rate)
        logger.info(f"Recording saved: {path} ({secs:.0f}s)")
        return path
