"""How long the agent waits before it decides the caller has finished: the turn pace.

An author picks a pace by name. Each framework turns it into its own floor and ceiling.
The numbers are Unmute's (unmute@868ae23 internal/target/pace.go paceProfiles).
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal, cast, get_args

from pydantic import AfterValidator, validate_call
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, named_errors, original_errors

TurnPace = Literal["snappy", "balanced", "patient"]
"""How eager the agent is to take its turn."""

TURN_PACES: tuple[str, ...] = get_args(TurnPace)
DEFAULT_PACE: TurnPace = "balanced"
"""The pace of an agent that names none: the faster behaviour, not a framework default."""

Framework = Literal["livekit", "pipecat"]


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class TurnWindow:
    """The quiet the agent waits before it takes its turn.

    Attributes:
        vad_silence: The floor: the silence the VAD waits before it says speech stopped and
            the end-of-turn model is asked.
        min_delay: The shortest wait LiveKit's endpointing considers.
        max_delay: The ceiling: the longest the agent waits for a caller the end-of-turn
            model thinks is not done.
        mode: LiveKit's endpointing mode. ``dynamic`` adapts inside the window to the
            pauses this caller leaves. Pipecat has no mode and reads the floor and ceiling.
    """

    vad_silence: float
    min_delay: float
    max_delay: float
    mode: Literal["fixed", "dynamic"] = "fixed"


TURN_WINDOWS: Mapping[Framework, Mapping[TurnPace, TurnWindow]] = MappingProxyType(
    {
        "livekit": MappingProxyType(
            {
                "snappy": TurnWindow(0.25, 0.25, 1.2, "dynamic"),
                "balanced": TurnWindow(0.3, 0.3, 1.6, "dynamic"),
                "patient": TurnWindow(0.4, 0.5, 2.5, "fixed"),
            }
        ),
        "pipecat": MappingProxyType(
            {
                "snappy": TurnWindow(0.2, 0.2, 1.2),
                "balanced": TurnWindow(0.2, 0.2, 1.6),
                "patient": TurnWindow(0.2, 0.2, 3.0),
            }
        ),
    }
)
"""Each framework's window for each pace.

On Pipecat the floor stays at 0.2 s for every pace. That is measured, not chosen: with the
SLNG speech-to-text bridge a wider floor makes a turn slower, so the pace moves only the
ceiling there. A Pipecat agent on another speech path may pass LiveKit's row instead.
"""


def _known_pace(pace: str | None) -> str | None:
    """Return the pace once it is one of the three, or None.

    Raises:
        ValueError: Naming the paces to use. The message names ``turn_pace``, so the
            caller can tell which setting to fix.
    """
    if pace is not None and pace not in TURN_PACES:
        raise ValueError(f"turn_pace {pace!r} is not one of {', '.join(TURN_PACES)}")
    return pace


@named_errors("turn_window")
@validate_call
def turn_window(
    pace: Annotated[str | None, AfterValidator(_known_pace)], framework: Framework = "livekit"
) -> TurnWindow:
    """Return a framework's window for a pace.

    Args:
        pace: ``snappy``, ``balanced`` or ``patient``. None is ``balanced``.
        framework: ``livekit`` or ``pipecat``.

    Returns:
        The floor and ceiling that framework uses for the pace.

    Raises:
        ValueError: If the pace is not one of the three, or the framework is not one of
            the two.
    """
    return TURN_WINDOWS[framework][cast(TurnPace, DEFAULT_PACE if pace is None else pace)]
