"""Replays of provider drop-outs seen on live calls (see agent/recovery.py)."""

import asyncio

import pytest

from agent.recovery import CallRecovery, context_id_from_error

CTX = "7153ea86-0297-46fc-ba72-bf51a2df0542"
NO_AUDIO = f"TTS context {CTX} completed with no audio"
STT_DOWN = "received 1011 (internal error) Deepgram did not receive audio data within the timeout window"


@pytest.fixture
def harness(tmp_path):
    spoken: list[str] = []
    incidents: list[int] = []
    lang = {"value": "en"}

    async def speak(text):
        spoken.append(text)

    recovery = CallRecovery(
        call_id="test",
        speak=speak,
        language=lambda: lang["value"],
        on_incident=lambda: incidents.append(1),
        log_path=tmp_path / "incidents.jsonl",
        stt_only_delay=0.02,
        reconnect_settle=0.01,
    )
    return recovery, spoken, incidents, lang


async def settle():
    await asyncio.sleep(0.08)


def drop(recovery, ctx, text):
    """The TTS was asked to say `text` in context `ctx`, and produced no audio."""
    recovery.on_tts_request(ctx, text)
    error = f"TTS context {ctx} completed with no audio"
    recovery.on_tts_no_audio(error, context_id_from_error(error))


def test_context_id_parsed_from_pipecat_error():
    assert context_id_from_error(NO_AUDIO) == CTX
    assert context_id_from_error("something else") is None


async def test_dropped_reply_waits_for_reconnect_then_is_spoken(harness):
    recovery, spoken, incidents, _ = harness
    recovery.on_tts_request(CTX, "Could you say that again, ")
    recovery.on_tts_request(CTX, "when will you pay?")  # token-streamed chunks
    recovery.on_tts_no_audio(NO_AUDIO, CTX)
    await settle()
    assert spoken == []  # nothing is sent into a dead connection
    recovery.on_tts_connected()
    await settle()
    assert spoken == ["Sorry, the line broke up for a second. Could you say that again, when will you pay?"]
    recovery.on_bot_started_speaking()
    assert len(incidents) == 1


async def test_live_network_switch_replay(harness):
    """22:50:58 network drop: idle prompt dropped, STT error, TTS back at :17.9, caller back at :23."""
    recovery, spoken, incidents, _ = harness
    recovery.on_tts_connected()  # initial connect at call start: not a reconnect
    drop(recovery, CTX, "Hello, are you still there?")  # 22:51:08
    recovery.on_stt_error(STT_DOWN)  # 22:51:11 (TTS still down: nothing is spoken yet)
    await settle()
    assert spoken == []
    recovery.on_tts_connected()  # 22:51:17.9 Cartesia reconnects
    await settle()
    assert spoken == ["Sorry, the line broke up for a second. Hello, are you still there?"]
    recovery.on_bot_started_speaking()
    assert len(incidents) == 1  # one outage, one incident


async def test_recovery_line_dropped_again_is_retried_on_next_reconnect(harness):
    recovery, spoken, _, _ = harness
    drop(recovery, CTX, "Hello?")
    recovery.on_tts_connected()
    await settle()
    assert len(spoken) == 1
    drop(recovery, "ctx-2", spoken[0])  # the TTS died again before it played
    await settle()
    assert len(spoken) == 1
    recovery.on_tts_connected()
    await settle()
    assert len(spoken) == 2 and spoken[1] == spoken[0]


async def test_stt_only_outage_asks_caller_to_repeat_in_their_language(harness):
    recovery, spoken, _, lang = harness
    lang["value"] = "hi"
    recovery.on_stt_error(STT_DOWN)
    await settle()
    assert spoken == ["माफ़ कीजिए, लाइन एक पल के लिए कट गई थी। क्या आप दोबारा बता सकते हैं?"]


async def test_caller_speaking_first_cancels_the_apology(harness):
    """Live call: the caller said "Hello?" first and the LLM answered naturally."""
    recovery, spoken, _, _ = harness
    drop(recovery, CTX, "Hello, are you still there?")
    recovery.on_user_started_speaking()
    recovery.on_tts_connected()
    await settle()
    assert spoken == []


async def test_agent_speaking_normally_cancels_the_apology(harness):
    recovery, spoken, _, _ = harness
    recovery.on_stt_error(STT_DOWN)
    recovery.on_bot_started_speaking()
    await settle()
    assert spoken == []


async def test_gives_up_after_bounded_attempts(harness):
    recovery, spoken, _, _ = harness
    drop(recovery, CTX, "Hello?")
    for i in range(5):
        recovery.on_tts_connected()
        await settle()
        if spoken:
            drop(recovery, f"ctx-{i}", spoken[-1])
    assert len(spoken) == 3
