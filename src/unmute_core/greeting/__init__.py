"""How a call opens: a fixed line, one the model words, or silence.

``Greeting(text=...)`` says a line, ``Greeting()`` has the model word one, and
``Greeting(speaks_first="user")`` waits for the caller.

``greet`` in ``unmute_core.livekit`` and ``greeting_frames`` in ``unmute_core.pipecat`` open
the call with it.
"""

from unmute_core.greeting._greeting import GREETING_HOLD_SECONDS, GREETING_INSTRUCTION, Greeting

__all__ = ["GREETING_HOLD_SECONDS", "GREETING_INSTRUCTION", "Greeting"]
