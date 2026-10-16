"""What the replies of one call on Pipecat show: their text, calls and the caller's turns.

``UnmuteFlows`` feeds it the frames the LLM processes and pushes. It knows nothing of
runs: a caller that needs the step a reply belongs to passes it in and gets it back.
"""

import asyncio
import contextlib
from collections.abc import Generator, Sequence
from dataclasses import replace
from typing import Any, cast

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pydantic import SkipValidation
from pydantic.dataclasses import dataclass as pydantic_dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.handoff import newest_caller
from unmute_core.pipecat._history import (
    STEP_START_MESSAGE,
    caller_turns,
    ends_with_tool_result,
    trailing_results,
)
from unmute_core.tasks import FINISH_TOOL_NAME, AfterReply, StepRun


@original_errors
@pydantic_dataclass(frozen=True, config=VALUE)
class _Playing:
    """The line a step's reply is playing, and whether the bot has started to speak it."""

    step: SkipValidation[StepRun]
    line: str
    started: bool = False


class ReplyTracker:
    """The text and calls of the newest reply, the request it answers and the caller's turns.

    Attributes:
        caller_count: Caller turns the call has held. It only grows, whatever a step's
            history keeps.
    """

    def __init__(self) -> None:
        """Start with no reply, no request and no caller turn."""
        self._reply: list[str] = []
        self._reply_step: object | None = None
        self._answers: list[tuple[str, str]] = []
        self._playing: _Playing | None = None
        self._called = False
        self._cut = False
        self._request: list[Any] = []
        self._caller_message: dict[str, Any] | None = None
        self.caller_count = 0
        self._calls: set[str] = set()
        self._line_played: asyncio.Event | None = None
        self._line_started = False

    @property
    def text(self) -> str:
        """The text of the newest reply: the one streaming now, or the last one."""
        return "".join(self._reply).strip()

    @property
    def request(self) -> list[Any]:
        """The messages of the newest model request."""
        return self._request

    @property
    def answers(self) -> list[tuple[str, str]]:
        """The tool results the newest model request answers, as ``(name, result)``."""
        return self._answers

    def alone(self, call_id: str) -> bool:
        """Return True when ``call_id`` is the only tool call of the newest reply."""
        return self._calls <= {call_id}

    def note_request(self, messages: Sequence[Any]) -> None:
        """Record a model request's messages, the tool results it answers and its caller turn."""
        self._request = list(messages)
        self._answers = trailing_results(self._request)
        self.count_caller(self._request)

    def count_caller(self, messages: Sequence[Any]) -> None:
        """Count a caller turn when ``messages`` hold a caller message no request held before.

        The step cue is no turn.
        """
        newest = newest_caller(messages, skip_cue=False)
        if newest is None or newest is self._caller_message or newest == STEP_START_MESSAGE:
            return
        self._caller_message = newest
        self.caller_count += 1

    def play(self, step: StepRun, line: str) -> None:
        """Note that ``step``'s reply line is being played."""
        self._playing = _Playing(step, line)

    @contextlib.contextmanager
    def awaiting_line(self) -> Generator[asyncio.Event]:
        """Yield an event set once the bot has said the line queued inside the block.

        The line is played when the bot stops speaking after it started, or an
        interruption cuts it off.
        """
        played = self._line_played = asyncio.Event()
        self._line_started = False
        try:
            yield played
        finally:
            self._line_played = None

    def observe(self, frame: Frame) -> None:
        """Note an interruption, and how a reply line played.

        A reply line is heard once the bot stops speaking after it started. An
        interruption before that cut it off, or dropped it before it began.
        """
        if isinstance(frame, InterruptionFrame):
            self._cut = True
        if (played := self._line_played) is not None:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._line_started = True
            elif isinstance(frame, InterruptionFrame) or (
                isinstance(frame, BotStoppedSpeakingFrame) and self._line_started
            ):
                played.set()
        if self._playing is None:
            return
        playing = self._playing
        if isinstance(frame, InterruptionFrame):
            self._playing = None
            playing.step.reply_played(playing.line, heard=False)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._playing = replace(playing, started=True)
        elif isinstance(frame, BotStoppedSpeakingFrame) and playing.started:
            self._playing = None
            playing.step.reply_played(playing.line, heard=True)

    def pushed[S](self, frame: Frame, step: S | None) -> S | None:
        """Follow a frame the LLM pushes: a reply's text and calls.

        Args:
            frame: The frame.
            step: The step running now, or None. A reply belongs to the step that ran
                when it started.

        Returns:
            The step whose reply just ended, or None.
        """
        if isinstance(frame, LLMFullResponseStartFrame):
            self._reply = []
            self._reply_step = step
            self._called = self._cut = False
        elif isinstance(frame, LLMTextFrame):
            self._reply.append(frame.text)
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._called = True
            self._calls = {call.tool_call_id for call in frame.function_calls}
        elif isinstance(frame, LLMFullResponseEndFrame):
            ended, self._reply_step = self._reply_step, None
            return cast(Any, ended)
        return None

    def finished(self, step: StepRun) -> AfterReply | None:
        """Tell ``step`` its reply ended.

        Returns:
            The finish to send after the reply, a held one or a finish check's, or None.
        """
        return step.reply_ended(
            spoke=bool(self.text),
            called=self._called,
            completed=not self._cut,
            answers_result=ends_with_tool_result(self._request),
            caller_turns=caller_turns(self._request),
        )

    def unheard(self, step: StepRun) -> tuple[str, str] | None:
        """Return the step tool result a finish answers without a word, or None."""
        if self.text:
            return None
        return step.unheard(
            self._answers, refused=any(name == FINISH_TOOL_NAME for name, _ in self._answers)
        )
