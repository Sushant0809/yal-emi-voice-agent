"""Optional per-call tracing to Langfuse, over OpenTelemetry.

    docker compose -f docker-compose.langfuse.yml up -d     # local Langfuse at http://localhost:3000
    # .env: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (the compose file pre-creates pk-lf-local / sk-lf-local)

One trace per call (Pipecat's ``conversation`` span): a ``turn`` span per exchange, with
the STT, LLM and TTS spans Pipecat records under it, plus a ``tool.<name>`` span per tool
call added here. Each trace is tagged with the prompt version, model, lender and language,
and carries a keyed hash of the borrower id (never the id itself) so calls to one borrower
group together. Its session id is the call id, the same id as the transcript, recording
and outcome record.

Every span passes through ``RedactingExporter`` before it leaves the process: DOBs, phone
digits, four-digit groups and borrower names are masked in every attribute, transcripts and
LLM messages included. ``TRACE_CONTENT=none`` drops conversation text altogether.

Tracing is optional and must never affect a call: with no keys it is off; spans are exported
on a background thread; an unreachable Langfuse only loses spans; every helper here swallows
its own errors.
"""

import base64
import contextlib
import hashlib
import hmac
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from loguru import logger

from agent.config import CONFIG
from agent.redact import redact, redact_text

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )

    OTEL_AVAILABLE = True
except ImportError:  # tracing is an optional extra
    OTEL_AVAILABLE = False
    SpanExporter = object  # type: ignore[assignment,misc]

TRACE_CONTENT = CONFIG.trace_content

# Attributes that carry conversation text (Pipecat's names, plus ours on tool spans).
CONTENT_KEYS = {
    "input", "output", "transcript", "text", "tools", "gen_ai.system_instructions",
    "langfuse.observation.input", "langfuse.observation.output",
}

_enabled = False


def enabled() -> bool:
    return _enabled


# --- Export-side redaction ------------------------------------------------------------------


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [_scrub(v) for v in value]
    return value


def scrub_attributes(attributes: Mapping[str, Any] | None, content: str = TRACE_CONTENT) -> dict[str, Any]:
    out = {}
    for key, value in (attributes or {}).items():
        if key in CONTENT_KEYS and content == "none":
            continue
        # Our own langfuse.* attributes are built from hashes, ids and config, never caller data.
        out[key] = value if key.startswith("langfuse.") and key not in CONTENT_KEYS else _scrub(value)
    return out


class RedactingExporter(SpanExporter):
    """Wraps the real exporter: each span is copied with scrubbed attributes and event
    attributes, so nothing unredacted is ever serialised. Export errors are contained."""

    def __init__(self, inner: "SpanExporter", content: str = TRACE_CONTENT):
        self.inner = inner
        self.content = content

    def _copy(self, s: "ReadableSpan") -> "ReadableSpan":
        events = [type(e)(e.name, scrub_attributes(e.attributes, self.content), e.timestamp) for e in s.events]
        return ReadableSpan(
            name=s.name, context=s.context, parent=s.parent, resource=s.resource,
            attributes=scrub_attributes(s.attributes, self.content), events=events, links=s.links,
            kind=s.kind, status=s.status, start_time=s.start_time, end_time=s.end_time,
            instrumentation_scope=s.instrumentation_scope,
        )

    def export(self, spans: "Sequence[ReadableSpan]") -> "SpanExportResult":
        try:
            return self.inner.export([self._copy(s) for s in spans])
        except Exception as e:
            logger.debug(f"Trace export failed (call unaffected): {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return self.inner.force_flush(timeout_millis)
        except Exception:
            return False


# --- Set-up -----------------------------------------------------------------------------------


def setup(exporter: "SpanExporter | None" = None) -> bool:
    """Turn tracing on if Langfuse keys are configured (or an exporter is given, for tests)."""
    global _enabled
    if _enabled:
        return True
    if not OTEL_AVAILABLE or (exporter is None and not (CONFIG.langfuse_public_key and CONFIG.langfuse_secret_key)):
        return False
    try:
        if exporter is None:
            auth = base64.b64encode(f"{CONFIG.langfuse_public_key}:{CONFIG.langfuse_secret_key}".encode()).decode()
            exporter = OTLPSpanExporter(
                endpoint=f"{CONFIG.langfuse_host}/api/public/otel/v1/traces",
                headers={"Authorization": f"Basic {auth}"},
                timeout=5,
            )
        provider = TracerProvider(resource=Resource.create({"service.name": "lumen-emi-agent"}))
        provider.add_span_processor(BatchSpanProcessor(RedactingExporter(exporter)))
        trace.set_tracer_provider(provider)
        _enabled = True
        logger.info(f"Tracing to Langfuse at {CONFIG.langfuse_host} (content: {TRACE_CONTENT})")
    except Exception as e:
        logger.warning(f"Tracing disabled, set-up failed: {e}")
    return _enabled


def flush(timeout_millis: int = 3000) -> None:
    """Push a finished call's spans out now rather than on the next batch tick."""
    if _enabled:
        with contextlib.suppress(Exception):
            trace.get_tracer_provider().force_flush(timeout_millis)


# --- Per-call attributes ----------------------------------------------------------------------


def hashed_borrower_id(borrower_id: str) -> str:
    """Keyed hash: stable per borrower, so their calls group together, but not reversible by
    hashing the (small, guessable) id space without the key."""
    key = (CONFIG.trace_id_salt or CONFIG.langfuse_secret_key or "lumen-dev").encode()
    return "b_" + hmac.new(key, borrower_id.encode(), hashlib.sha256).hexdigest()[:16]


def call_attributes(*, call_id: str, borrower_id: str, prompt_version: str, model: str,
                    lender: str, language: str) -> dict[str, Any]:
    """Trace-level attributes for Langfuse, set on the conversation span (the trace root)."""
    return {
        "langfuse.trace.name": "emi-reminder-call",
        "langfuse.session.id": call_id,
        "langfuse.user.id": hashed_borrower_id(borrower_id),
        "langfuse.version": prompt_version,
        "langfuse.trace.tags": [f"prompt:{prompt_version}", f"model:{model}", f"lender:{lender}",
                                f"lang:{language}"],
        "langfuse.trace.metadata.call_id": call_id,
        "langfuse.trace.metadata.prompt_version": prompt_version,
    }


def _turn_parent(worker: Any):
    ctx = getattr(worker, "_tracing_context", None)  # Pipecat's per-pipeline tracing context
    return (ctx.get_turn_context() or ctx.get_conversation_context()) if ctx else None


@contextlib.contextmanager
def tool_span(worker: Any, name: str, args: dict) -> Iterator[Any]:
    """A span for one tool call, under the current turn. Yields a callable that records the
    result. A no-op when tracing is off; never raises into the tool call."""
    if not _enabled:
        yield lambda result: None
        return
    span = None
    try:
        span = trace.get_tracer("lumen.agent").start_span(f"tool.{name}", context=_turn_parent(worker))
        span.set_attribute("langfuse.observation.type", "tool")
        span.set_attribute("langfuse.observation.input", json.dumps(redact(args), ensure_ascii=False))
    except Exception as e:
        logger.debug(f"tool span failed to start: {e}")

    def record(result: dict) -> None:
        with contextlib.suppress(Exception):
            if span is not None:
                span.set_attribute("langfuse.observation.output", json.dumps(redact(result), ensure_ascii=False))
                if "error" in result:
                    span.set_attribute("langfuse.observation.level", "ERROR")

    try:
        yield record
    finally:
        with contextlib.suppress(Exception):
            if span is not None:
                span.end()


def annotate_outcome(worker: Any, outcome: dict | None) -> None:
    """Put the call's outcome on the trace (as trace output and filterable metadata), so
    Langfuse can be searched by outcome and online evals can find, say, every hardship call."""
    if not _enabled or not outcome:
        return
    try:
        attrs = {
            "langfuse.trace.output": json.dumps(
                {k: outcome.get(k) for k in ("outcome", "verified", "escalate_to_human", "payment_link_sent",
                                             "promise_to_pay_date", "language", "turns", "duration_secs")}),
            "langfuse.trace.metadata.outcome": str(outcome.get("outcome")),
            "langfuse.trace.metadata.verified": str(outcome.get("verified")),
            "langfuse.trace.metadata.escalate_to_human": str(outcome.get("escalate_to_human")),
        }
        observer = getattr(worker, "turn_trace_observer", None)
        root = getattr(observer, "_conversation_span", None)
        if root is not None and root.is_recording():
            root.set_attributes(attrs)
        else:  # conversation already closed: a child span carries the same trace-level attributes
            ctx = getattr(worker, "_tracing_context", None)
            parent = ctx.get_conversation_context() if ctx else None
            span = trace.get_tracer("lumen.agent").start_span("call.outcome", context=parent)
            span.set_attributes(attrs)
            span.end()
    except Exception as e:
        logger.debug(f"Could not annotate trace with outcome: {e}")
