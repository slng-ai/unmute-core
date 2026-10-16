"""The rule names for speech over the agent, shared by both frameworks."""

from typing import Literal

Backchannels = Literal["keep", "drop", "stop"]
"""What a short "mm-hm" or "yes" over the agent does.

The same rule names on LiveKit and Pipecat. See ``unmute_core.livekit.interruption`` and
``unmute_core.pipecat.interruption``.
"""
