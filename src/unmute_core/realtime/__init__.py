"""Realtime and live models: what they can do on any framework, and the texts they read.

A realtime model hears the caller, thinks and speaks over one connection to its own
server. No request passes through the library, so ``ModelCall`` and the model hooks never
reach it. The plug point is the framework's own realtime model. Each adapter adds one
small helper: ``unmute_core.livekit`` for any LiveKit ``RealtimeModel``, and
``unmute_core.pipecat`` for OpenAI's realtime and live services.
"""

from unmute_core.realtime._rules import (
    MODEL_HOOKS,
    SAY_LINE,
    SETTLE_WAIT_SECONDS,
    check_realtime,
    check_realtime_handoff,
    realtime_instructions,
    say_line,
)

__all__ = [
    "MODEL_HOOKS",
    "SAY_LINE",
    "SETTLE_WAIT_SECONDS",
    "check_realtime",
    "check_realtime_handoff",
    "realtime_instructions",
    "say_line",
]
