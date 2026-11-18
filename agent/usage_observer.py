"""Logs each LLM request's token usage (cache hits, reasoning tokens) per call.

Answers two latency questions from live calls that the latency breakdown can't:
did the prompt cache hit, and did the model spend tokens reasoning before speaking?
"""

import json
from pathlib import Path

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed


class LLMUsageObserver(BaseObserver):
    def __init__(self, call_id: str, path: Path, **kwargs):
        super().__init__(**kwargs)
        self._call_id = call_id
        self._path = path
        self._seen: set[int] = set()  # a frame is observed at every hop; log it once

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if not isinstance(frame, MetricsFrame) or frame.id in self._seen:
            return
        self._seen.add(frame.id)
        for m in frame.data:
            if isinstance(m, LLMUsageMetricsData):
                u = m.value
                row = {
                    "call_id": self._call_id,
                    "model": m.model,
                    "prompt_tokens": u.prompt_tokens,
                    "cache_read": u.cache_read_input_tokens,
                    "cache_write": u.cache_creation_input_tokens,
                    "completion_tokens": u.completion_tokens,
                    "reasoning_tokens": u.reasoning_tokens,
                }
                with open(self._path, "a") as f:
                    f.write(json.dumps(row) + "\n")
