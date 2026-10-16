"""A turn pace as LiveKit's endpointing options.

As Unmute's emitted LiveKit code applies it (unmute@868ae23 internal/generate/templates/
livekit_v1/call.py.tmpl): the floor is the prewarmed Silero VAD's silence window, and the
ceiling is the session's endpointing.
"""

from livekit.agents.voice.turn import EndpointingOptions

from unmute_core.turn import TurnWindow


def endpointing(window: TurnWindow) -> EndpointingOptions:
    """Return the ceiling of a window as ``TurnHandlingOptions(endpointing=...)``.

    Load the VAD with the floor too: ``silero.VAD.load(min_silence_duration=window.vad_silence)``.

    Args:
        window: From ``unmute_core.turn.turn_window(pace, "livekit")``.
    """
    return {"mode": window.mode, "min_delay": window.min_delay, "max_delay": window.max_delay}
