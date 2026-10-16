"""The turn pace: how long the agent waits before it decides the caller has finished.

``turn_window("snappy", "livekit")`` gives the floor and ceiling one framework uses for a
pace. ``unmute_core.livekit.endpointing``, and ``unmute_core.pipecat.vad_params`` and
``smart_turn_params``, turn the window into that framework's own options.
"""

from unmute_core.turn._backchannels import Backchannels
from unmute_core.turn._interruption import InterruptionRule, word_count
from unmute_core.turn._pace import (
    DEFAULT_PACE,
    TURN_PACES,
    TURN_WINDOWS,
    TurnPace,
    TurnWindow,
    turn_window,
)

__all__ = [
    "DEFAULT_PACE",
    "TURN_PACES",
    "TURN_WINDOWS",
    "Backchannels",
    "InterruptionRule",
    "TurnPace",
    "TurnWindow",
    "turn_window",
    "word_count",
]
