"""Speech-to-text accuracy on what matters in these calls: digits, dates and amounts.

    uv run python -m evals.wer                  # Flux, the live STT, on every case
    uv run python -m evals.wer --engine nova    # Nova-3 (the STT_ENGINE=nova fallback)
    uv run python -m evals.wer --lang hi

Each case in evals/wer_cases.yaml is a sentence a borrower might say, in English, Hindi or
Hinglish. Its audio is synthesised once with Cartesia and cached in results/wer/audio/, or
taken from a real recording (`audio:` in the case). Flux is streaming-only, so the clip is
played in real time through Pipecat's DeepgramFluxSTTService with the live bot's settings;
Nova-3 uses Deepgram's file API.

Two numbers per language:
- WER: word error rate against the reference (lower-cased, punctuation and Devanagari
  nukta/chandrabindu variants normalised), raw and "normalised". The normalised WER also maps
  month names and number words to one token in either script, because Flux often writes
  "December" for "दिसंबर" in a Hindi sentence, which is right. Both still penalise "12,500" for
  "twelve thousand five hundred", so read WER as a rough guide only.
- Entity accuracy: did each date, digit string or amount survive, in any spelling (digits,
  English words, Hindi words)? This is what decides whether verification and the promise
  date work, and is the number to watch.

Caveat: synthesised speech is cleaner than a caller on a phone line, so these are upper
bounds. Add real recordings to the cases to measure real accuracy.
"""

import argparse
import asyncio
import hashlib
import io
import os
import re
import unicodedata
import wave
from datetime import date, datetime
from pathlib import Path

import httpx
import numpy as np
import yaml

from agent.config import CONFIG, ROOT
from agent.speech import (
    _HI_1_99,
    _HI_MONTHS,
    date_to_words,
    hindi_number_to_words,
    number_to_words,
)

CASES_PATH = ROOT / "evals" / "wer_cases.yaml"
OUT_DIR = ROOT / "results" / "wer"
SAMPLE_RATE = 16000
EN_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
             "september", "october", "november", "december"]


# --- Normalisation and matching ---------------------------------------------------------------


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = t.replace("़", "").replace("ँ", "ं")  # nukta; chandrabindu -> anusvara
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 12,500 -> 12500
    t = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", t)  # 27th -> 27
    t = re.sub(r"[.,!?;:।\"'()\[\]\-–—/]", " ", t)  # not \W: it would strip Devanagari vowel signs
    return " ".join(SPELLINGS.get(w, w) for w in t.split())


# Equally correct Hindi spellings that STT and our references pick differently.
SPELLINGS = {"रुपए": "रुपये", "उनासी": "उन्यासी"}


def _n(text: str) -> str:
    return normalise(text)


DIGIT_WORDS = {
    **{w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())},
    "oh": "0", "o": "0",
    **{_n(w): str(i) for i, w in enumerate("शून्य एक दो तीन चार पाँच छह सात आठ नौ".split())},
    _n("पांच"): "5", _n("छः"): "6", _n("छे"): "6",
}


def digit_runs(hyp: str) -> list[str]:
    """Digit strings in a transcript, joining spoken digits and digit groups ("two three 45")."""
    runs, current = [], ""
    for tok in normalise(hyp).split():
        d = tok if tok.isdigit() else DIGIT_WORDS.get(tok)
        if d is None:
            if current:
                runs.append(current)
            current = ""
        else:
            current += d
    return runs + ([current] if current else [])


def amount_forms(v: int) -> set[str]:
    return {str(v), _n(number_to_words(v)), _n(hindi_number_to_words(v))}


def day_forms(d: int) -> set[str]:
    ordinal = date_to_words(date(2000, 1, d)).split()[1]  # "the fifth of January" -> "fifth"
    return {str(d), _n(ordinal), _n(number_to_words(d)), _n(hindi_number_to_words(d))}


def year_forms(y: int) -> set[str]:
    hi, lo = divmod(y, 100)
    return {str(y), _n(f"{number_to_words(hi)} {number_to_words(lo)}"),
            _n(f"{hindi_number_to_words(hi)} सौ {hindi_number_to_words(lo)}"), _n(hindi_number_to_words(y))}


def _has(hyp: str, forms: set[str]) -> bool:
    padded = f" {hyp} "
    return any(f" {f} " in padded for f in forms)


def entity_found(entity: dict, hypothesis: str) -> bool:
    hyp = " ".join(t for t in normalise(hypothesis).split() if t != "and")
    if "digits" in entity:
        return any(entity["digits"] in run for run in digit_runs(hypothesis))
    if "amount" in entity:
        return _has(hyp, amount_forms(int(entity["amount"])))
    if "day" in entity:
        return _has(hyp, day_forms(int(entity["day"])))
    year, month, day = (entity["date"].split("-") if not entity["date"].startswith("--")
                        else (None, *entity["date"][2:].split("-")))
    month_forms = {EN_MONTHS[int(month) - 1], _n(_HI_MONTHS[int(month) - 1])}
    ok = _has(hyp, day_forms(int(day))) and _has(hyp, month_forms)
    return ok and (year is None or _has(hyp, year_forms(int(year))))


def _canonical_words() -> dict[str, str]:
    """Same meaning, either script: month names and number words up to 99 map to one token."""
    words = {}
    for i, (en, hi) in enumerate(zip(EN_MONTHS, _HI_MONTHS, strict=True), 1):
        words[en] = words[_n(hi)] = f"<month{i}>"
    for n in range(100):
        en = _n(number_to_words(n))
        if " " not in en:  # single-token forms only ("fourteen", "ninety")
            words[en] = f"<{n}>"
    for n, hi in enumerate(["शून्य", *_HI_1_99]):
        words[_n(hi)] = f"<{n}>"
    words.update({"number": _n("नंबर"), "rupees": _n("रुपये")})
    return words


CANONICAL = _canonical_words()


def entity_type(entity: dict) -> str:
    return next(iter(entity))


def wer(reference: str, hypothesis: str, canonical: bool = False) -> float:
    """Word error rate. canonical=True first maps month names and number words in either script
    to one token, so "December" for "दिसंबर" or "fourteen" for "चौदह" isn't counted as an error."""
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    if canonical:
        ref, hyp = [CANONICAL.get(w, w) for w in ref], [CANONICAL.get(w, w) for w in hyp]
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / max(len(ref), 1)


# --- Audio --------------------------------------------------------------------------------------


def read_wav(path: Path) -> bytes:
    """16 kHz mono PCM from any WAV (real recordings can be stereo or 48 kHz)."""
    with wave.open(str(path)) as w:
        rate, channels = w.getframerate(), w.getnchannels()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        t = np.arange(0, len(samples) / rate, 1 / SAMPLE_RATE)
        samples = np.interp(t, np.arange(len(samples)) / rate, samples)
    return samples.astype(np.int16).tobytes()


async def synthesise(http: httpx.AsyncClient, case: dict) -> Path:
    """Cartesia audio for the case, cached by text and voice."""
    key = hashlib.sha256(f"{case['text']}|{CONFIG.cartesia_voice_id}".encode()).hexdigest()[:8]
    path = OUT_DIR / "audio" / f"{case['id']}-{key}.wav"
    if path.exists():
        return path
    r = await http.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={"X-API-Key": os.environ["CARTESIA_API_KEY"], "Cartesia-Version": "2026-03-01"},
        json={"model_id": "sonic-3", "transcript": case["text"],
              "voice": {"mode": "id", "id": CONFIG.cartesia_voice_id},
              "language": "en" if case["lang"] == "en" else "hi",
              "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE}},
        timeout=30,
    )
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return path


def _wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


# --- Transcription ------------------------------------------------------------------------------


async def transcribe_nova(http: httpx.AsyncClient, pcm: bytes) -> str:
    r = await http.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": CONFIG.deepgram_model, "language": CONFIG.deepgram_language, "punctuate": "true"},
        headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}", "Content-Type": "audio/wav"},
        content=_wav_bytes(pcm), timeout=60,
    )
    r.raise_for_status()
    return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


async def transcribe_flux(pcm: bytes) -> str:
    """Play the clip in real time through the live bot's Flux STT and collect final transcripts."""
    from pipecat.frames.frames import EndFrame, InputAudioRawFrame, TranscriptionFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
    from pipecat.transcriptions.language import Language
    from pipecat.workers.runner import WorkerRunner

    texts: list[str] = []

    class Collect(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                texts.append(frame.text)
            await self.push_frame(frame, direction)

    stt = DeepgramFluxSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramFluxSTTService.Settings(
            model=CONFIG.flux_model, language_hints=[Language.EN, Language.HI],
            eager_eot_threshold=CONFIG.flux_eager_eot_threshold, eot_threshold=CONFIG.flux_eot_threshold,
        ),
    )
    worker = PipelineWorker(Pipeline([stt, Collect()]), params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE),
                            enable_turn_tracking=False)

    async def feed():
        await asyncio.sleep(1.5)  # let the websocket connect
        chunk = int(SAMPLE_RATE * 0.02) * 2  # 20 ms
        audio = pcm + bytes(SAMPLE_RATE * 2 * 2)  # 2 s of silence so Flux closes the turn
        for i in range(0, len(audio), chunk):
            await worker.queue_frame(InputAudioRawFrame(audio=audio[i:i + chunk], sample_rate=SAMPLE_RATE, num_channels=1))
            await asyncio.sleep(0.02)
        for _ in range(30):
            if texts:
                break
            await asyncio.sleep(0.1)
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await asyncio.gather(runner.run(), feed())
    return " ".join(texts)


# --- Report -------------------------------------------------------------------------------------


def score(case: dict, hypothesis: str) -> dict:
    ents = [{"type": entity_type(e), "value": str(next(iter(e.values()))), "found": entity_found(e, hypothesis)}
            for e in case.get("entities", [])]
    return {"id": case["id"], "lang": case["lang"], "reference": case["text"], "hypothesis": hypothesis,
            "wer": wer(case["text"], hypothesis), "wer_norm": wer(case["text"], hypothesis, canonical=True),
            "entities": ents, "real_audio": bool(case.get("audio"))}


def report(results: list[dict], engine: str) -> str:
    lines = [f"# STT accuracy: {engine}", "",
             "| Language | Clips | WER (raw) | WER (normalised) | Entities correct | Digits | Dates | Amounts |",
             "|---|---|---|---|---|---|---|---|"]
    for lang in ("en", "hi", "mixed", "all"):
        rs = [r for r in results if lang in ("all", r["lang"])]
        if not rs:
            continue
        ents = [e for r in rs for e in r["entities"]]

        def acc(kinds, ents=ents):
            sel = [e for e in ents if e["type"] in kinds]
            return f"{sum(e['found'] for e in sel)}/{len(sel)}" if sel else "-"

        name = {"en": "English", "hi": "Hindi", "mixed": "Hinglish", "all": "**All**"}[lang]
        lines.append(f"| {name} | {len(rs)} | {sum(r['wer'] for r in rs) / len(rs):.0%} "
                     f"| {sum(r['wer_norm'] for r in rs) / len(rs):.0%} | {acc({'digits', 'date', 'day', 'amount'})} "
                     f"| {acc({'digits'})} | {acc({'date', 'day'})} | {acc({'amount'})} |")
    misses = [(r, e) for r in results for e in r["entities"] if not e["found"]]
    lines += ["", f"## Entity misses ({len(misses)})", ""]
    lines += [f"- **{r['id']}** {e['type']} {e['value']}: heard \"{r['hypothesis']}\"" for r, e in misses] or ["None."]
    synthetic = sum(not r["real_audio"] for r in results)
    lines += ["", f"{synthetic} of {len(results)} clips are synthesised speech (Cartesia), so these are upper bounds."]
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["flux", "nova"], default="flux")
    parser.add_argument("--lang", choices=["en", "hi", "mixed"])
    parser.add_argument("-c", "--concurrency", type=int, default=5)
    args = parser.parse_args()

    cases = yaml.safe_load(CASES_PATH.read_text())["cases"]
    if args.lang:
        cases = [c for c in cases if c["lang"] == args.lang]
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as http:
        async def one(case):
            async with sem:
                path = ROOT / case["audio"] if case.get("audio") else await synthesise(http, case)
                pcm = read_wav(path)
                hyp = await (transcribe_flux(pcm) if args.engine == "flux" else transcribe_nova(http, pcm))
                return score(case, hyp)

        results = await asyncio.gather(*map(one, cases))

    text = report(results, f"{args.engine} ({CONFIG.flux_model if args.engine == 'flux' else CONFIG.deepgram_model})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.engine}.md"
    out.write_text(text + "\n## All clips\n\n" + "\n".join(
        f"- {r['id']} (WER {r['wer']:.0%}): \"{r['reference']}\" → \"{r['hypothesis']}\"" for r in results) + "\n")
    print(text + f"\nSaved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
