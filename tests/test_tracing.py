"""Tracing: PII never leaves the process, tags are right, and nothing here can break a call."""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent import tracing
from agent.prompts import prompt_version
from agent.tools import REGISTRY, Tool, register

SAID = "My date of birth is 14/03/1990, I'm Ananya Iyer, mobile ending 2345."


def spans_through(exporter, content="redacted", **attrs):
    """Record one span with `attrs` through a RedactingExporter; return what was exported."""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(tracing.RedactingExporter(exporter, content)))
    span = provider.get_tracer("t").start_span("stt")
    span.set_attributes(attrs)
    span.add_event("said", {"text": SAID})
    span.end()
    return exporter.get_finished_spans() if hasattr(exporter, "get_finished_spans") else None


def test_pii_is_masked_in_every_attribute_and_event():
    attrs = {"transcript": SAID, "input": json.dumps([{"role": "user", "content": SAID}]),
             "langfuse.user.id": "b_abc", "langfuse.trace.tags": ["prompt:x", "lang:en"]}
    [span] = spans_through(InMemorySpanExporter(), **attrs)
    exported = json.dumps(dict(span.attributes)) + json.dumps(dict(span.events[0].attributes))
    for secret in ("1990", "2345", "Ananya", "Iyer"):
        assert secret not in exported
    assert "[redacted]" in span.attributes["transcript"] and "[name]" in span.attributes["transcript"]
    assert span.attributes["langfuse.user.id"] == "b_abc"  # our own attributes pass untouched
    assert list(span.attributes["langfuse.trace.tags"]) == ["prompt:x", "lang:en"]


def test_trace_content_none_drops_conversation_text():
    [span] = spans_through(InMemorySpanExporter(), content="none", transcript=SAID, **{"metrics.ttfb": 0.4})
    assert "transcript" not in span.attributes and span.attributes["metrics.ttfb"] == 0.4


def test_a_failing_exporter_is_contained():
    class Broken:
        def export(self, spans):
            raise ConnectionError("langfuse down")

    out = tracing.RedactingExporter(Broken()).export([])
    assert out is SpanExportResult.FAILURE


def test_unreachable_langfuse_does_not_raise_or_hang():
    exporter = tracing.RedactingExporter(OTLPSpanExporter(endpoint="http://127.0.0.1:9/v1/traces", timeout=1))
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    t0 = time.monotonic()
    provider.get_tracer("t").start_span("turn").end()
    assert time.monotonic() - t0 < 5


def test_otlp_request_reaches_langfuse_endpoint_with_auth_and_no_pii():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers["Authorization"]
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    host = f"http://127.0.0.1:{server.server_port}"
    auth = base64.b64encode(b"pk-lf-local:sk-lf-local").decode()
    inner = OTLPSpanExporter(endpoint=f"{host}/api/public/otel/v1/traces",
                             headers={"Authorization": f"Basic {auth}"}, timeout=5)
    spans_through(inner, transcript=SAID)
    server.server_close()
    assert seen["path"] == "/api/public/otel/v1/traces" and seen["auth"] == f"Basic {auth}"
    assert b"1990" not in seen["body"] and b"Ananya" not in seen["body"] and b"[redacted]" in seen["body"]


def test_setup_is_off_without_keys(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.setattr(tracing, "CONFIG", SimpleNamespace(langfuse_public_key="", langfuse_secret_key=""))
    assert tracing.setup() is False and not tracing.enabled()


def test_helpers_are_no_ops_when_tracing_is_off(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", False)
    with tracing.tool_span(object(), "verify_identity", {"dob": "1990-03-14"}) as record:
        record({"verified": True})
    tracing.annotate_outcome(object(), {"outcome": "hardship"})
    tracing.flush()


# --- With a live (in-memory) provider ---------------------------------------------------------


@pytest.fixture
def traced(monkeypatch):
    """Route the global tracer into memory for one test."""
    mem = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(tracing.RedactingExporter(mem)))
    monkeypatch.setattr(tracing.trace, "get_tracer", provider.get_tracer)
    monkeypatch.setattr(tracing, "_enabled", True)
    return provider, mem


def test_tool_span_under_the_current_turn_with_redacted_input(traced):
    provider, mem = traced
    turn = provider.get_tracer("pipecat").start_span("turn")
    ctx = SimpleNamespace(get_turn_context=lambda: tracing.trace.set_span_in_context(turn),
                          get_conversation_context=lambda: None)
    worker = SimpleNamespace(_tracing_context=ctx)
    with tracing.tool_span(worker, "verify_identity", {"dob": "1990-03-14"}) as record:
        record({"verified": False, "error": "Tool failed"})
    turn.end()
    tool = next(s for s in mem.get_finished_spans() if s.name == "tool.verify_identity")
    assert tool.parent.span_id == turn.get_span_context().span_id
    assert "1990" not in tool.attributes["langfuse.observation.input"]
    assert tool.attributes["langfuse.observation.level"] == "ERROR"


def test_outcome_lands_on_the_conversation_span(traced):
    provider, mem = traced
    root = provider.get_tracer("pipecat").start_span("conversation")
    worker = SimpleNamespace(turn_trace_observer=SimpleNamespace(_conversation_span=root), _tracing_context=None)
    tracing.annotate_outcome(worker, {"outcome": "hardship", "verified": True, "escalate_to_human": True})
    root.end()
    [span] = mem.get_finished_spans()
    assert span.attributes["langfuse.trace.metadata.outcome"] == "hardship"
    assert json.loads(span.attributes["langfuse.trace.output"])["escalate_to_human"] is True


# --- Trace attributes ----------------------------------------------------------------------


def test_call_attributes_tag_the_trace_without_the_raw_borrower_id():
    attrs = tracing.call_attributes(call_id="c1", borrower_id="LF-1001", prompt_version="abc123",
                                    model="claude-sonnet-5-5", lender="lumen", language="en")
    assert attrs["langfuse.session.id"] == "c1" and attrs["langfuse.version"] == "abc123"
    assert "LF-1001" not in json.dumps(attrs)
    assert attrs["langfuse.user.id"] == tracing.hashed_borrower_id("LF-1001")  # stable per borrower
    assert tracing.hashed_borrower_id("LF-1001") != tracing.hashed_borrower_id("LF-1002")
    assert "prompt:abc123" in attrs["langfuse.trace.tags"]


def test_prompt_version_changes_when_a_tool_description_changes():
    before = prompt_version()
    assert before == prompt_version()  # deterministic

    async def handler(state, args):
        return {}

    register(Tool(name="demo_tool", description="Demo.", properties={}, required=[], handler=handler))
    try:
        assert prompt_version() != before
    finally:
        REGISTRY.pop("demo_tool")
    assert prompt_version() == before
