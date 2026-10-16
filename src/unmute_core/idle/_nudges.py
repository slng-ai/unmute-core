"""Speak to a caller who went quiet, then hang up: up to three lines, each after its wait.

Every event goes to one ``on_event``, and ``speak`` takes only the text. A nudge with no
text is worded by the model, the second nudge is optional, and so is the farewell line.
"""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from enum import IntEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+", re.UNICODE)

GENERATED_NUDGE_INSTRUCTIONS = "The caller went quiet. Briefly check whether they are still there."
"""What the model is told for a nudge with no text (Unmute's inactivity nudge)."""

Line = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
"""A line code speaks: trimmed, never blank."""

Wait = Annotated[float, Field(ge=1)]
"""Seconds of quiet before a line, at least one."""


class Stage(IntEnum):
    """Which line of an idle sequence comes next. The value indexes the three waits."""

    FIRST = 0
    SECOND = 1
    HANGUP = 2


class IdleNudges(BaseModel):
    """What to say to a quiet caller, and when.

    Each wait starts when the caller and the agent are both quiet: after the agent's last
    line, or after the previous nudge has played. Any word from the caller, or any reply
    from the agent, starts again from the first nudge.

    Attributes:
        enabled: Off keeps every timer still.
        first_nudge_delay_seconds: Quiet before the first nudge. None gives no nudges: the
            call ends after ``hangup_delay_seconds`` of quiet.
        second_nudge_delay_seconds: Quiet after the first nudge, before the second. None
            gives no second nudge: the hang-up wait follows the first.
        hangup_delay_seconds: Quiet after the last nudge, before the call ends. None never
            ends the call: the agent only nudges.
        first_nudge_text: The first nudge, like "Are you still there?". None lets the model
            word it.
        second_nudge_text: The second nudge. None lets the model word it.
        final_hangup_text: Said just before the call ends. None ends it without a line.
        nudge_instruction: What the model is told for a nudge with no text. Write it in
            the language the model should answer in, or leave the English default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    first_nudge_delay_seconds: Wait | None = 15
    second_nudge_delay_seconds: Wait | None = 30
    hangup_delay_seconds: Wait | None = 15
    first_nudge_text: Line | None = None
    second_nudge_text: Line | None = None
    final_hangup_text: Line | None = None
    nudge_instruction: Line = GENERATED_NUDGE_INSTRUCTIONS

    @model_validator(mode="after")
    def _lines_need_their_waits(self) -> Self:
        """Refuse a line with no wait before it, and a config with nothing to do."""
        if self.first_nudge_delay_seconds is None and self.hangup_delay_seconds is None:
            raise ValueError("no nudge and no hang-up: use enabled=False instead")
        if self.first_nudge_delay_seconds is None and (
            self.second_nudge_delay_seconds is not None
            or self.first_nudge_text is not None
            or self.second_nudge_text is not None
        ):
            raise ValueError(
                "no first nudge: set second_nudge_delay_seconds=None and leave out nudge texts"
            )
        if self.second_nudge_delay_seconds is None and self.second_nudge_text is not None:
            raise ValueError("a second nudge text needs a second nudge delay")
        if self.hangup_delay_seconds is None and self.final_hangup_text is not None:
            raise ValueError("a final hang-up text needs a hang-up delay")
        return self

    @property
    def worded_by_model(self) -> bool:
        """True when a nudge has no text, so the adapter must give a way to generate one."""
        return (self.first_nudge_delay_seconds is not None and self.first_nudge_text is None) or (
            self.second_nudge_delay_seconds is not None and self.second_nudge_text is None
        )


class IdleNudgeController:
    """Run ``IdleNudges`` for one call, from the agent's state and the caller's words.

    The adapter reports each agent state change and each caller transcript. Speaking
    stops the timer and starts the stages over. Thinking holds the timer and keeps the
    stage, so a slow tool or model request never draws a nudge. Listening after speech
    starts the wait for the current stage.
    """

    def __init__(
        self,
        config: IdleNudges | None,
        *,
        speak: Callable[[str], Awaitable[bool]],
        hangup: Callable[[], Awaitable[object]],
        on_event: Callable[[str, dict[str, Any] | None], None] | None = None,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
        generate: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        """Build the controller. It does nothing until the agent first stops speaking.

        Args:
            config: The lines and waits. None, or ``enabled=False``, turns it off.
            speak: Say a line and wait for it to end. Return True when the caller cut in.
            hangup: End the call.
            on_event: Called with each ``idle_nudges.*`` event, for the trace.
            sleep: Wait a number of seconds. Tests replace it.
            generate: Have the model reply to instructions and wait for the reply to end.
                Return True when the caller cut in. Needed for a nudge with no text.

        Raises:
            ValueError: If a nudge has no text and there is no ``generate``.
        """
        self._config = config if config is not None and config.enabled else None
        if self._config is not None and self._config.worded_by_model and generate is None:
            raise ValueError("a nudge with no text is worded by the model: pass generate")
        self._generate = generate
        self._speak = speak
        self._hangup = hangup
        self._on_event = on_event
        self._sleep = sleep
        self._first = (
            Stage.FIRST
            if self._config is None or self._config.first_nudge_delay_seconds
            else Stage.HANGUP
        )
        self._stage: Stage = self._first
        self._sequence = 0
        self._task: asyncio.Task[None] | None = None
        self._awaiting_agent_response = False
        self._speaking_nudge = False
        self._suspended = 0
        self._agent_state: str | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        """Return True when there is something to say."""
        return self._config is not None

    def close(self) -> None:
        """Stop for good, once the call ends."""
        self._closed = True
        self._suspended = 0
        self._cancel_task()

    def on_agent_state_changed(self, old_state: str | None, new_state: str | None) -> None:
        """Follow the agent: "speaking", "thinking" or "listening"."""
        if self._config is None or self._closed:
            return
        old = (old_state or "").strip().lower()
        new = (new_state or "").strip().lower()
        self._agent_state = new or None
        if new == "speaking":
            if not self._speaking_nudge:
                self._invalidate("assistant_speaking", reset_stage=True)
                self._awaiting_agent_response = False
            return
        if new == "thinking":
            if self._speaking_nudge:
                return
            self._invalidate("assistant_thinking", reset_stage=False)
            return
        if new != "listening" or self._speaking_nudge:
            return
        if old == "speaking":
            self._awaiting_agent_response = False
            self._arm("assistant_playout_complete")
        elif self._awaiting_agent_response:
            self._awaiting_agent_response = False
            self._arm("user_turn_complete")

    def on_user_transcript(self, transcript: str | None) -> None:
        """Start over on any word from the caller, interim transcripts included."""
        if self._config is None or self._closed:
            return
        if not transcript or _WORD.search(transcript) is None:
            return
        already_reset = (
            self._awaiting_agent_response
            and self._stage == self._first
            and (self._task is None or self._task.done())
        )
        self._awaiting_agent_response = True
        self._invalidate("user_transcript", reset_stage=True)
        if not already_reset:
            self._emit("idle_nudges.reset", {"reason": "user_transcript"})

    def suspend(self, reason: str) -> None:
        """Hold every timer until the matching ``resume``. Calls nest."""
        if self._config is None or self._closed:
            return
        self._suspended += 1
        if self._suspended == 1:
            self._sequence += 1
            self._cancel_task()
            self._emit("idle_nudges.suspended", {"reason": reason, "stage": self._stage.value})

    def resume(self, reason: str) -> None:
        """Release one ``suspend``. The last one starts the wait again while listening."""
        if self._config is None or self._closed or self._suspended <= 0:
            return
        self._suspended -= 1
        if self._suspended:
            return
        self._emit("idle_nudges.resumed", {"reason": reason, "stage": self._stage.value})
        if self._agent_state == "listening":
            self._arm("resumed")

    def _cancel_task(self) -> None:
        """Stop the running wait, if any."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def _invalidate(self, reason: str, *, reset_stage: bool) -> None:
        """Make any running wait stale, and optionally start over at the first nudge."""
        self._sequence += 1
        if reset_stage:
            self._stage = self._first
        if not self._speaking_nudge:
            self._cancel_task()
        self._emit("idle_nudges.canceled", {"reason": reason, "stage": self._stage.value})

    def _arm(self, reason: str) -> None:
        """Start the wait for the current stage."""
        if self._config is None or self._closed or self._suspended:
            return
        if self._stage is Stage.HANGUP and self._config.hangup_delay_seconds is None:
            return
        self._cancel_task()
        delay = self._delay(self._stage)
        self._emit(
            "idle_nudges.armed",
            {"reason": reason, "stage": self._stage.value, "delay_seconds": delay},
        )
        self._task = asyncio.create_task(self._run_stage(self._sequence, self._stage))

    def _delay(self, stage: Stage) -> float:
        """Return the wait before a stage's line."""
        config = self._config
        if config is None:
            raise RuntimeError("idle nudges are off, so no stage has a wait")
        return (
            config.first_nudge_delay_seconds or 0,
            config.second_nudge_delay_seconds or 0,
            config.hangup_delay_seconds or 0,
        )[stage]

    async def _run_stage(self, sequence: int, stage: Stage) -> None:
        """Run one stage. A line that cannot be said or a failed hang-up only logs."""
        try:
            await self._stage_line(sequence, stage)
        except Exception as exc:
            logger.warning("idle nudge stage %d failed", stage, exc_info=True)
            self._emit("idle_nudges.error", {"stage": stage.value, "error": str(exc)})

    async def _stage_line(self, sequence: int, stage: Stage) -> None:
        """Wait, say the stage's line, then arm the next stage or hang up."""
        await self._sleep(self._delay(stage))
        if not self._is_current(sequence, stage) or self._config is None:
            return
        name, text = (
            ("first", self._config.first_nudge_text),
            ("second", self._config.second_nudge_text),
            ("final_message", self._config.final_hangup_text),
        )[stage]
        if stage is not Stage.HANGUP or text is not None:
            self._emit(f"idle_nudges.{name}.fired", {"text": text} if text else {"generated": True})
            interrupted = await self._say(text)
            if interrupted or not self._is_current(sequence, stage):
                return
        if stage is not Stage.HANGUP:
            self._stage = (
                Stage.SECOND
                if stage is Stage.FIRST and self._config.second_nudge_delay_seconds
                else Stage.HANGUP
            )
            self._arm(f"{name}_nudge_playout_complete")
            return
        self._emit("idle_nudges.hangup", None)
        await self._hangup()

    async def _say(self, text: str | None) -> bool:
        """Say a nudge's line, or have the model word one. Return True when cut in."""
        self._speaking_nudge = True
        try:
            if text is not None:
                return await self._speak(text)
            if self._generate is None or self._config is None:
                raise RuntimeError("a nudge with no text is worded by the model: pass generate")
            return await self._generate(self._config.nudge_instruction)
        finally:
            self._speaking_nudge = False

    def _is_current(self, sequence: int, stage: Stage) -> bool:
        """Return True while nothing has happened since this wait started."""
        return not self._closed and self._sequence == sequence and self._stage == stage

    def _emit(self, event: str, data: dict[str, Any] | None) -> None:
        """Report an event, and log it."""
        logger.debug("%s %s", event, data)
        if self._on_event is not None:
            self._on_event(event, data)
