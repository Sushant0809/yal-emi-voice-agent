"""Lumen Finance EMI reminder voice agent (Pipecat).

Pipeline:  mic -> VAD + Smart Turn -> Deepgram STT -> Claude -> speech formatter -> Cartesia TTS -> speaker

Run:  uv run python -m agent.bot     then open http://localhost:7860
"""

import asyncio
import json
import os
import time

from anthropic import AsyncAnthropic
from dotenv import dotenv_values
from loguru import logger
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndWorkerFrame,
    FunctionCallResultProperties,
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from agent.borrowers import get_borrower
from agent.call_state import CallState, Outcome
from agent.config import CONFIG, ROOT, model_supports_effort
from agent.prompts import build_system_prompt, first_message_instruction
from agent import templates
from agent.phrases import lender
from agent.recovery import BotSpeechObserver, CallRecovery, context_id_from_error
from agent.speech import detect_language, spoken_transform
from agent import transcripts
from agent.recording import CallRecorder
from agent.tools import REGISTRY, closing_line, run_tool, tool_specs
from agent.usage_observer import LLMUsageObserver

# Fixed lines (fillers, idle prompts, recovery) come from the lender's phrase pack:
# config/lenders/<LENDER>/<lang>.yaml. Validated at import, so a bad pack fails at start-up.
LENDER = lender()
# Fillers aren't the agent's own words (used to decide whether it already said goodbye).
FILLER_KEYS = {t.filler for t in REGISTRY.values() if t.filler} | {"fillers.verify_identity"}
FILLER_LINES = set().union(*(LENDER.all_lines(key) for key in FILLER_KEYS))


# --- Warm start (process level) -------------------------------------------------
# One Anthropic client for the whole process: its HTTP connection pool (TCP + TLS)
# stays open between calls instead of being rebuilt for every call.
ANTHROPIC = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or None)  # None -> `ant auth login`

# Load the on-device models (Silero VAD, Smart Turn v3) once at server start, so the
# ONNX weights are already read from disk when the first call connects.
_t0 = time.perf_counter()
SileroVADAnalyzer()
LocalSmartTurnAnalyzerV3()
logger.info(f"Preloaded VAD + Smart Turn in {time.perf_counter() - _t0:.2f}s")


def _llm_settings(system_prompt: str) -> AnthropicLLMService.Settings:
    settings = AnthropicLLMService.Settings(
        model=CONFIG.agent_model,
        system_instruction=system_prompt,
        enable_prompt_caching=CONFIG.prompt_caching,
    )
    if model_supports_effort(CONFIG.agent_model):
        settings.effort = CONFIG.agent_effort  # else Pipecat sends the model's fastest default
    if CONFIG.agent_thinking == "between_tools":
        settings.thinking = AnthropicLLMService.ThinkingConfig(type="between_tools")
    return settings


async def prewarm_llm(llm: AnthropicLLMService, context: LLMContext, system_prompt: str) -> None:
    """Fill the prompt cache (tools + system) and open the connection before the greeting.

    A ``max_tokens=0`` request runs prefill only: it writes the cache entry and returns
    immediately with no output tokens. The tools/system blocks come from Pipecat's own
    adapter so they byte-match what the real requests send, which a cache hit requires.
    """
    if not CONFIG.prompt_caching:
        return
    t0 = time.perf_counter()
    try:
        # Pipecat appends its own sections (e.g. "ASYNC TOOLS") to the system prompt when tools
        # are registered. Warm the cache with exactly that composed prompt: warming the base
        # prompt alone missed the cache, so the first LLM reply of each call ran cold (~2s).
        composed = getattr(llm._settings, "system_instruction", None) or system_prompt
        params = await llm.get_llm_adapter().get_llm_invocation_params(
            context, enable_prompt_caching=True, system_instruction=composed
        )
        extra = {"thinking": {"type": "between_tools"}} if CONFIG.agent_thinking == "between_tools" else {}
        if model_supports_effort(CONFIG.agent_model):
            extra["output_config"] = {"effort": CONFIG.agent_effort}
        response = await ANTHROPIC.messages.create(
            model=CONFIG.agent_model,
            max_tokens=0,
            system=params["system"],
            tools=params["tools"],
            messages=[{"role": "user", "content": "(call connecting)"}],
            **extra,
        )
        u = response.usage
        logger.info(
            f"LLM prewarm {time.perf_counter() - t0:.2f}s: cache write {u.cache_creation_input_tokens}, "
            f"cache read {u.cache_read_input_tokens}"
        )
    except Exception as e:  # warming is best-effort; the call works without it
        logger.warning(f"LLM prewarm failed: {e}")


def _build_stt():
    if CONFIG.stt_engine == "flux":
        # Flux predicts end-of-turn early (EagerEndOfTurn); Pipecat starts the LLM on that
        # prediction and holds the response until the turn is confirmed, discarding it if the
        # caller resumes speaking.
        return DeepgramFluxSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            enable_eager_end_of_turn=True,
            settings=DeepgramFluxSTTService.Settings(
                model=CONFIG.flux_model,
                language_hints=[Language.EN, Language.HI],
                eager_eot_threshold=CONFIG.flux_eager_eot_threshold,
                eot_threshold=CONFIG.flux_eot_threshold,
            ),
        )
    return DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model=CONFIG.deepgram_model,
            language=CONFIG.deepgram_language,  # "multi" = English/Hindi code-switching
        ),
    )


def _turn_strategies() -> UserTurnStrategies | None:
    """Smart Turn for Nova; None for Flux, which recommends its own (eager) strategies.
    Passing ours with Flux would override them and run two turn detectors at once."""
    if CONFIG.stt_engine == "flux":
        return None
    return UserTurnStrategies(
        stop=[
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(
                    params=SmartTurnParams(stop_secs=CONFIG.smart_turn_stop_secs)
                )
            )
        ]
    )


def _borrower_id(runner_args: RunnerArguments) -> str:
    body = getattr(runner_args, "body", None)
    if isinstance(body, dict) and body.get("borrower_id"):
        return body["borrower_id"]
    # Re-read .env on every call so the borrower can be switched without a restart.
    return dotenv_values(ROOT / ".env").get("DEFAULT_BORROWER_ID") or CONFIG.default_borrower_id


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    state = CallState(borrower=get_borrower(_borrower_id(runner_args)))
    state.language = state.borrower.preferred_language
    logger.info(f"Calling {state.borrower.borrower_id} ({state.borrower.name}), call {state.call_id}")

    stt = _build_stt()
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        # SENTENCE (default): whole sentences go to Cartesia, so prosody is natural. TOKEN streamed
        # single tokens and Cartesia started on "Thank" alone, stretching it (heard live and
        # measured in a recording: "Thank" lasted 0.88s, split from "you, Ananya").
        text_aggregation_mode=TextAggregationMode(CONFIG.tts_text_mode),
        # Server-side buffer only matters when streaming tokens. In sentence mode Pipecat already
        # sends whole sentences, and None lets it disable Cartesia's buffer (no double buffering).
        max_buffer_delay_ms=CONFIG.cartesia_max_buffer_ms if CONFIG.tts_text_mode == "token" else None,
        text_transforms=[("*", spoken_transform)],
        settings=CartesiaTTSService.Settings(voice=CONFIG.cartesia_voice_id or None),
    )
    system_prompt = build_system_prompt(state.borrower)
    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        client=ANTHROPIC,  # shared, already-warm connection pool
        settings=_llm_settings(system_prompt),
    )

    # What the agent has said since the caller's last turn (to avoid a double goodbye).
    spoke_this_turn = {"text": ""}
    speaking_language = {"value": state.borrower.preferred_language}

    # Tools: one Pipecat handler that delegates to the framework-free run_tool().
    async def handle_tool(params: FunctionCallParams):
        result = await run_tool(state, params.function_name, dict(params.arguments))
        logger.info(f"tool {params.function_name}({dict(params.arguments)}) -> {result}")
        if params.function_name == "verify_identity":
            line = templates.reminder(state.borrower, result, speaking_language["value"]) if CONFIG.templated_turns else None
            if line:
                # Verified, no history, English: speak the reminder now, skip the second LLM pass.
                await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
                await worker.queue_frames([TTSSpeakFrame(line)])
                return
            filler = LENDER.phrase(speaking_language["value"], "fillers.verify_identity")
            await worker.queue_frames([TTSSpeakFrame(filler)])  # covers the LLM pass that follows
        if REGISTRY[params.function_name].ends_call:
            if line := closing_line(dict(params.arguments), spoke_this_turn["text"], state.language):
                await params.llm.push_frame(TTSSpeakFrame(line))
            await params.result_callback(result, properties=FunctionCallResultProperties(run_llm=False))
            await asyncio.to_thread(state.save)
            logger.info(f"Outcome: {json.dumps(state.outcome.model_dump(mode='json'))}")
            # The goodbye was queued ahead of this; EndWorkerFrame flushes it, then hangs up.
            await params.llm.push_frame(EndWorkerFrame())
            return
        await params.result_callback(result)

    # Every registered tool, no per-tool wiring. All are synchronous (cancel_on_interruption):
    # registering one as async makes Pipecat append an "ASYNC TOOLS" section to the system
    # prompt at pipeline start, after the prewarm, so the first LLM turn of every call missed
    # the cache (~2s, heard live as "Thank you... [pause]"). The outcome is recorded before
    # the goodbye plays and is saved again on disconnect, so a barge-in can't lose it.
    for t in REGISTRY.values():
        llm.register_function(t.name, handle_tool, cancel_on_interruption=True)

    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=s["name"],
                description=s["description"],
                properties=s["properties"],
                required=s["required"],
            )
            for s in tool_specs()
        ]
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        for call in function_calls:
            tool = REGISTRY.get(call.function_name)
            if tool and tool.filler:  # declared by the tool; covers its run time
                filler = LENDER.phrase(speaking_language["value"], tool.filler)
                await worker.queue_frames([TTSSpeakFrame(filler)])

    context = LLMContext(tools=tools)
    # Runs while WebRTC is still connecting, so the greeting hits a warm cache.
    prewarm = asyncio.create_task(prewarm_llm(llm, context, system_prompt))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # Who decides the caller has finished: Flux's own turn model (with speculative
            # early generation), or Smart Turn v3 with a 1.2s silence fallback for Nova.
            user_turn_strategies=_turn_strategies(),
            user_idle_timeout=CONFIG.user_idle_timeout_secs,
        ),
    )

    # Whole-call audio (caller left, Asha right) -> outcomes/recordings/<call_id>.wav
    recorder = CallRecorder(state.call_id) if CONFIG.record_calls else None

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            *([recorder.processor] if recorder else []),  # after output: hears both sides
            assistant_aggregator,
        ]
    )

    # Recovery from provider drop-outs (see agent/recovery.py).
    recovery = CallRecovery(
        call_id=state.call_id,
        speak=lambda text: worker.queue_frames([TTSSpeakFrame(text)]),
        language=lambda: speaking_language["value"],
        on_incident=state.record_incident,
        log_path=CONFIG.outcomes_dir / "incidents.jsonl",
    )

    latency = UserBotLatencyObserver()
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[
            latency,
            LLMUsageObserver(state.call_id, CONFIG.outcomes_dir / "llm_usage.jsonl"),
            BotSpeechObserver(recovery),
        ],
    )

    @tts.event_handler("on_tts_request")
    async def on_tts_request(service, context_id, text):
        recovery.on_tts_request(context_id, text)
        # Fillers ("Thanks, one moment") aren't the agent's own words: counting them made
        # end_call skip the goodbye field, and a live call hung up right after a filler.
        if text.strip() not in FILLER_LINES:
            spoke_this_turn["text"] += " " + text
        # Fillers and recovery lines follow the language Asha is speaking right now, so they
        # never switch language mid-conversation (seen live: English filler in a Hindi call).
        if lang := detect_language(text):
            speaking_language["value"] = lang

    @tts.event_handler("on_connected")
    async def on_tts_connected(service):
        recovery.on_tts_connected()  # fires on reconnect too; that's when a pending line is spoken

    @user_aggregator.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy):
        recovery.on_user_started_speaking()

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame):
        if frame.processor is tts and "no audio" in frame.error:
            recovery.on_tts_no_audio(frame.error, context_id_from_error(frame.error))
        elif frame.processor is stt:
            recovery.on_stt_error(frame.error)
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    latency_log = CONFIG.outcomes_dir / "latency.jsonl"
    latency_log.parent.mkdir(parents=True, exist_ok=True)

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        with open(latency_log, "a") as f:
            f.write(json.dumps({"call_id": state.call_id, **breakdown.model_dump(mode="json")}) + "\n")
        for line in breakdown.turn_contribution_lines():
            logger.debug(f"latency {line}")

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        state.turns += 1
        state.idle_prompts = 0
        spoke_this_turn["text"] = ""
        # Track the language the caller is actually using (fillers and the outcome record use it).
        # One- or two-word replies ("Payment", "OK") don't count, mirroring the prompt rule.
        if message.content and len(message.content.split()) >= 3 and (lang := detect_language(message.content)):
            state.language = lang

    @user_aggregator.event_handler("on_user_turn_idle")
    async def on_user_turn_idle(aggregator):
        if state.ended:
            return
        if state.idle_prompts < CONFIG.max_idle_prompts:
            prompts = LENDER.phrases(speaking_language["value"], "idle_prompts")
            text = prompts[min(state.idle_prompts, len(prompts) - 1)]
            state.idle_prompts += 1
            await worker.queue_frames([TTSSpeakFrame(text)])
            return
        state.finalize(Outcome.NO_RESPONSE, notes="Caller went silent; ended after idle prompts.")
        await asyncio.to_thread(state.save)
        await worker.queue_frames([TTSSpeakFrame(LENDER.phrase(speaking_language["value"], "idle_goodbye")), EndWorkerFrame()])

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        if CONFIG.templated_turns:
            # Greeting from a template: plays as soon as the call connects, no LLM wait.
            note = templates.CALL_CONNECTED_NOTE.format(name=state.borrower.name)
            context.add_message({"role": "developer", "content": note})
            await worker.queue_frames([TTSSpeakFrame(templates.greeting(state.borrower))])
            return
        context.add_message({"role": "developer", "content": first_message_instruction(state.borrower)})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        if not state.ended:
            state.finalize(Outcome.DISCONNECTED, notes="Caller hung up before the call was closed.")
        # Always save (it's idempotent): an outcome can be final yet unsaved if the end_call
        # path failed partway, as happened live when end_call raised after finalizing.
        await asyncio.to_thread(state.save)
        await runner.cancel()

    try:
        await runner.run()
    finally:
        prewarm.cancel()
        recovery.cancel()
        await asyncio.to_thread(state.save)  # belt and braces; no-op if already saved
        if recorder:
            try:
                await recorder.save()
            except Exception as e:  # never let recording break call teardown
                logger.warning(f"Could not save recording: {e}")
        try:
            outcome = state.outcome.model_dump(mode="json") if state.outcome else None
            path = transcripts.save(state.call_id, context.get_messages(), outcome)
            logger.info(f"Transcript saved: {path}")
        except Exception as e:  # never let transcript saving break call teardown
            logger.warning(f"Could not save transcript: {e}")


_file_log_added = False


def _ensure_file_log() -> None:
    """Debug log to a file too. Added here, not at import: Pipecat's runner calls
    logger.remove() on start-up, which silently dropped a sink added at import time."""
    global _file_log_added
    if not _file_log_added:
        CONFIG.outcomes_dir.mkdir(parents=True, exist_ok=True)
        logger.add(CONFIG.outcomes_dir / "bot.log", level="DEBUG", rotation="20 MB")
        _file_log_added = True


async def bot(runner_args: RunnerArguments):
    _ensure_file_log()
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                # Clean background noise (traffic, fans, TV) before VAD and Deepgram hear it,
                # so it neither garbles transcripts nor falsely interrupts Asha.
                audio_in_filter=RNNoiseFilter() if CONFIG.noise_filter == "rnnoise" else None,
            )
        },
    )
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    import agent.web  # noqa: F401  registers /borrowers (the borrower picker) on the runner's app

    main()
