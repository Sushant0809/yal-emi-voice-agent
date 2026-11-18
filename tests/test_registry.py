"""The tool registry: one generic verification gate, and new tools need no prompt or bot change."""

from agent.borrowers import get_borrower
from agent.call_state import CallState
from agent.prompts import build_system_prompt
from agent.tools import REGISTRY, Tool, register, run_tool, tool_specs


async def test_every_gated_tool_is_refused_before_verification(tmp_path):
    state = CallState(borrower=get_borrower("LF-1008"), history_db=tmp_path / "h.db")
    gated = [t.name for t in REGISTRY.values() if t.requires_verification]
    assert {"send_payment_link", "check_payment_status"} <= set(gated)
    for name in gated:
        assert "error" in await run_tool(state, name, {})


async def test_payment_status_after_verification(tmp_path):
    state = CallState(borrower=get_borrower("LF-1008"), history_db=tmp_path / "h.db")
    await run_tool(state, "verify_identity", {"phone_last4": "0011"})
    result = await run_tool(state, "check_payment_status", {})
    latest = result["recent_payments"][0]
    assert latest["status"] == "processing" and latest["amount_spoken"] == "seven thousand two hundred rupees"


def test_prompt_never_names_tools():
    prompt = build_system_prompt(get_borrower("LF-1001"))
    assert not [t for t in REGISTRY if t in prompt]


async def test_new_tool_plugs_in_without_prompt_or_bot_changes(tmp_path):
    async def handler(state, args):
        return {"ok": True}

    register(Tool(name="demo_tool", description="Demo.", properties={}, required=[],
                  handler=handler, requires_verification=True))
    try:
        assert "demo_tool" in [s["name"] for s in tool_specs()]  # offered to the model
        state = CallState(borrower=get_borrower("LF-1001"), history_db=tmp_path / "h.db")
        assert "error" in await run_tool(state, "demo_tool", {})  # gate applies automatically
        await run_tool(state, "verify_identity", {"phone_last4": "2345"})
        assert await run_tool(state, "demo_tool", {}) == {"ok": True}
    finally:
        REGISTRY.pop("demo_tool")


async def test_injected_failure_uses_error_path_then_recovers():
    from agent.borrowers import get_borrower
    from agent.call_state import CallState
    from agent.tools import run_tool

    state = CallState(borrower=get_borrower("LF-1002"), verified=True,
                      inject_failures={"send_payment_link": 1})
    first = await run_tool(state, "send_payment_link", {})
    assert "error" in first and not state.payment_link_sent
    second = await run_tool(state, "send_payment_link", {})
    assert second.get("sent") is True and state.payment_link_sent
