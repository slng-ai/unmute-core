"""The turn lifecycle both adapters follow: open a ``turn`` span, mark it answered, end it."""

from collections.abc import Sequence
from typing import TypedDict, Unpack

from opentelemetry.trace import Span

from unmute_core.telemetry._telemetry import Telemetry


class TurnEnd(TypedDict, total=False):
    """What an adapter tells ``Telemetry.end_turn`` when a turn ends."""

    user_text: str | None
    agent_lines: Sequence[str]
    interrupted: bool
    e2e: float | None
    stt_final: float | None
    eou: float | None
    llm_ttft: float | None
    tts_ttfb: float | None
    error: str | None


class TurnTracker:
    """The open ``turn`` span of one call, and whether the agent has answered in it.

    A turn is the caller's until the agent does something in it. The adapter decides when a
    caller's words start a turn and what the turn's text and latencies are. The tracker
    numbers the turns and keeps the span.

    Attributes:
        span: The open turn's span, or None between turns.
        answered: True once the agent did something in the open turn.
    """

    def __init__(self, telemetry: Telemetry) -> None:
        """Start with no turn open.

        Args:
            telemetry: The call's telemetry, which starts and ends the spans.
        """
        self._telemetry = telemetry
        self._count = 0
        self.span: Span | None = None
        self.answered = False

    @property
    def caller_open(self) -> bool:
        """True while a turn is open and nothing the agent did has answered in it."""
        return self.span is not None and not self.answered

    def start(self) -> Span:
        """Open the next turn under the call. End the earlier one first."""
        self._count += 1
        self.span = self._telemetry.start_turn(self._count)
        self.answered = False
        return self.span

    def open(self, *, answering: bool = True) -> Span:
        """Return the open turn, opening one when none is, for something the agent does.

        Args:
            answering: True marks the turn answered. False leaves it as it is.
        """
        span = self.span if self.span is not None else self.start()
        self.answered = self.answered or answering
        return span

    def end(self, **fields: Unpack[TurnEnd]) -> None:
        """End the open turn with what the adapter gathered. Nothing happens without one."""
        if self.span is None:
            return
        self._telemetry.end_turn(self.span, **fields)
        self.span = None
