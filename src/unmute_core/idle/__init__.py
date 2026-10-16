"""Speak to a caller who went quiet, then hang up; and end a call that runs too long.

``IdleNudges`` holds up to three lines and the quiet before each; a nudge with no text is
worded by the model. ``IdleNudgeController`` runs them for one call from the agent's state
and the caller's words. ``unmute_core.livekit`` and ``unmute_core.pipecat`` wire it to a
session. ``CallLimit`` ends a call after its max duration.
"""

from unmute_core.idle._limit import CallLimit
from unmute_core.idle._nudges import GENERATED_NUDGE_INSTRUCTIONS, IdleNudgeController, IdleNudges

__all__ = ["GENERATED_NUDGE_INSTRUCTIONS", "CallLimit", "IdleNudgeController", "IdleNudges"]
