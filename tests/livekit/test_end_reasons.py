"""Why a LiveKit call ended: its close reasons and each place that ends the call on purpose."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from unmute_core.call import (
    CALLER_HUNG_UP,
    ENDED,
    ERROR,
    IDLE,
    CallHooks,
)
from unmute_core.livekit import (
    idle_nudges,
)
from unmute_core.livekit._trace import end_reason


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
