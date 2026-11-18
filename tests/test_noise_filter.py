"""RNNoise filter sanity: it removes stationary noise and is fast enough for real time."""

import time

import numpy as np
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter


async def test_rnnoise_suppresses_noise_quickly():
    rate = 16000
    f = RNNoiseFilter()
    await f.start(rate)
    rng = np.random.default_rng(0)
    noise = (rng.normal(0, 2000, rate * 2)).astype(np.int16)  # 2s of loud hiss
    chunks, t0 = [], time.perf_counter()
    for i in range(0, len(noise), 320):  # 20ms frames, like the transport
        chunks.append(await f.filter(noise[i:i + 320].tobytes()))
    per_frame_ms = (time.perf_counter() - t0) / (len(noise) / 320) * 1000
    out = np.frombuffer(b"".join(chunks), dtype=np.int16)
    tail = out[len(out) // 2:]  # skip warm-up
    reduction_db = 20 * np.log10(np.std(noise) / max(np.std(tail), 1))
    print(f"noise reduction {reduction_db:.1f} dB, {per_frame_ms:.2f} ms per 20ms frame")
    assert reduction_db > 10
    assert per_frame_ms < 5  # well within real time
