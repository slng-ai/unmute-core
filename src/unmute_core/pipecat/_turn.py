"""A turn pace as Pipecat's VAD and smart-turn parameters.

As Unmute's emitted Pipecat code applies it (unmute@868ae23 internal/generate/templates/
pipecat_v1/call.py.tmpl): the floor is the Silero VAD's ``stop_secs``, and the ceiling is
the smart-turn analyzer's own ``stop_secs``, the longest a turn may run unfinished.
"""

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams

from unmute_core.turn import TurnWindow


def vad_params(window: TurnWindow) -> VADParams:
    """Return the floor of a window, for ``SileroVADAnalyzer(params=...)``.

    Args:
        window: From ``unmute_core.turn.turn_window(pace, "pipecat")``.
    """
    return VADParams(stop_secs=window.vad_silence)


def smart_turn_params(window: TurnWindow) -> SmartTurnParams:
    """Return the ceiling of a window, for ``LocalSmartTurnAnalyzerV3(params=...)``.

    Args:
        window: From ``unmute_core.turn.turn_window(pace, "pipecat")``.
    """
    return SmartTurnParams(stop_secs=window.max_delay)
