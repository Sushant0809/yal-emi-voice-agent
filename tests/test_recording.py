import wave

from agent import recording
from agent.recording import CallRecorder


async def test_call_recorder_writes_stereo_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(recording, "RECORDINGS_DIR", tmp_path)
    rec = CallRecorder("test-call")
    one_second = b"\x01\x00\x02\x00" * 24000  # stereo 16-bit frames at 24 kHz
    # Same path Pipecat uses when the buffer flushes at the end of a call
    await rec.processor._call_event_handler("on_audio_data", one_second, 24000, 2)
    await rec.processor._call_event_handler("on_audio_data", one_second, 24000, 2)
    path = await rec.save()
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (2, 24000, 2)
        assert w.getnframes() == 48000  # two seconds


async def test_no_audio_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(recording, "RECORDINGS_DIR", tmp_path)
    assert await CallRecorder("silent").save() is None
