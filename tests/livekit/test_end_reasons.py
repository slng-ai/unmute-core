"""Why a LiveKit call ended: its close reasons and each place that ends the call on purpose."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.livekit.fake import until
from tests.outputs import AnyResult
from unmute_core.call import (
    AGENT_ENDED,
    CALLER_HUNG_UP,
    ENDED,
    ERROR,
    IDLE,
    MAX_DURATION,
    CallHooks,
)
from unmute_core.livekit import (
    function_tools,
    idle_nudges,
    max_duration,
)
from unmute_core.livekit._trace import end_reason
from unmute_core.tools import ToolRegistry, ToolSpec

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}


@pytest.mark.parametrize(
    ("close", "reason"),
    [
        ("participant_disconnected", CALLER_HUNG_UP),
        ("error", ERROR),
        ("user_initiated", ENDED),
        ("job_shutdown", ENDED),
    ],
)
def test_livekits_close_reasons_become_the_shared_end_reasons(close: str, reason: str) -> None:
    assert end_reason(close) == reason


async def test_the_max_duration_asks_the_call_to_end_for_that_reason() -> None:
    session = SimpleNamespace(shut=0, handlers={})
    session.shutdown = lambda: setattr(session, "shut", session.shut + 1)
    session.on = lambda name, handler: None
    hooks = CallHooks()
    max_duration(cast(Any, session), 0.01, call=hooks)
    await until(lambda: session.shut == 1)
    assert (await hooks.end("ended")).reason == MAX_DURATION


async def test_the_idle_hang_up_asks_the_call_to_end_for_that_reason() -> None:
    handlers: dict[str, Any] = {}
    session = SimpleNamespace(shut=0)
    session.shutdown = lambda: setattr(session, "shut", session.shut + 1)
    session.on = lambda name, handler: handlers.setdefault(name, handler)
    hooks = CallHooks()
    controller = idle_nudges(cast(Any, session), None, call=hooks)
    await controller._hangup()
    assert session.shut == 1
    assert (await hooks.end("ended")).reason == IDLE


async def test_the_end_tool_of_function_tools_asks_the_call_to_end_for_that_reason() -> None:
    class Done:
        async def run(self, arguments: dict[str, Any], state: object) -> object:
            return {"done": True}

    spec = ToolSpec(
        "hang_up",
        "End the call.",
        {"type": "object", "properties": {}},
        AnyResult,
        Done(),
        effect="ends_conversation",
    )
    hooks = CallHooks()
    [tool] = function_tools(ToolRegistry([spec]), None, hooks=hooks)
    session = SimpleNamespace(shut=0)
    session.shutdown = lambda: setattr(session, "shut", session.shut + 1)
    session.once = lambda name, handler: None
    context = SimpleNamespace(
        session=session, function_call=SimpleNamespace(name="hang_up", call_id="c1")
    )
    await tool({}, context)
    assert session.shut == 1
    assert (await hooks.end("ended")).reason == AGENT_ENDED
