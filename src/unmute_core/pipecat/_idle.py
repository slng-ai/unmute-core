"""Run ``IdleNudges`` on a Pipecat pipeline, from the frames that pass through it.

The agent's state comes from the frames: the bot
speaks between ``BotStartedSpeakingFrame`` and ``BotStoppedSpeakingFrame``, and thinks
from a model response's start, and while a tool call runs, until it speaks or the
response ends with nothing to say.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any, cast

from openai.types.chat import ChatCompletionDeveloperMessageParam
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

from unmute_core.call import IDLE, CallHooks
from unmute_core.idle import IdleNudgeController, IdleNudges
from unmute_core.pipecat._frames import FrameTable
from unmute_core.pipecat._realtime import ask, follow_replies, speak, speaks_itself

PLAYOUT_TIMEOUT_SECONDS = 30.0
"""The longest a nudge waits to be heard before the controller goes on."""


class IdleNudgeObserver(BaseObserver):
    """Say each nudge after its quiet, then the final line, then end the call.

    Attributes:
        controller: The call's controller, to ``suspend`` and ``resume`` it around
            something that must not draw a nudge.
    """

    def __init__(
        self,
        config: IdleNudges | None,
        worker: PipelineWorker,
        *,
        hangup: Callable[[], Awaitable[object]] | None = None,
        on_event: Callable[[str, dict[str, Any] | None], None] | None = None,
        llm: LLMService[Any] | None = None,
        call: CallHooks | None = None,
    ) -> None:
        """Watch one call's pipeline. Add it to the worker's observers.

        Args:
            config: The lines and waits. None turns it off.
            worker: The call's worker, which speaks the lines and ends the call.
            hangup: How to end the call, in place of ``worker.end()``. That drains the
                pipeline first, so the final line plays.
            on_event: Called with each ``idle_nudges.*`` event, for the trace.
            llm: The call's LLM service. OpenAI's realtime or live service that speaks
                itself says the lines and words the nudges in replies of its own.
            call: The call's hooks. The hang-up is noted as ``IDLE`` before it runs.
        """
        super().__init__(observe_every_push=False)
        self._worker = worker
        self._model = llm if llm is not None and speaks_itself(llm) else None
        if self._model is not None:
            follow_replies(self._model)
        end = hangup or worker.end

        async def hang_up() -> None:
            """Note the end as idle, then end the call."""
            if call is not None:
                call.end_requested(IDLE)
            await end()

        self._state = "listening"
        self._calls = 0
        self._played = asyncio.Event()
        self.controller = IdleNudgeController(
            config,
            speak=self._speak,
            hangup=hang_up,
            on_event=on_event,
            generate=self._generate,
        )

    async def on_push_frame(self, data: FramePushed) -> None:
        """Turn a frame into an agent state or a caller's words."""
        if (
            data.frame.broadcast_sibling_id is not None
            and data.direction == FrameDirection.UPSTREAM
        ):
            return
        if (handler := _FRAMES.handler(data.frame)) is not None:
            handler(self, data)

    def _on_bot_started(self, data: FramePushed) -> None:
        """The bot speaks."""
        self._played.clear()
        self._set("speaking")

    def _on_bot_stopped(self, data: FramePushed) -> None:
        """The bot has played what it had."""
        self._played.set()
        self._set("thinking" if self._calls else "listening")

    def _on_response_start(self, data: FramePushed) -> None:
        """A model reply starts: the agent thinks, unless it is speaking."""
        if self._state != "speaking":
            self._set("thinking")

    def _on_response_end(self, data: FramePushed) -> None:
        """A model reply ends with nothing to say: the agent waits on its tools or the caller."""
        if self._state == "thinking":
            self._set("thinking" if self._calls else "listening")

    def _on_tool_started(self, data: FramePushed) -> None:
        """A tool call runs: the agent thinks, unless it is speaking."""
        self._calls += 1
        if self._state != "speaking":
            self._set("thinking")

    def _on_tool_ended(self, data: FramePushed) -> None:
        """A tool call returned or was cancelled."""
        self._calls = max(0, self._calls - 1)

    def _on_transcript(self, data: FramePushed) -> None:
        """The caller's words, so far or final."""
        self.controller.on_user_transcript(
            cast(TranscriptionFrame | InterimTranscriptionFrame, data.frame).text
        )

    def _on_end(self, data: FramePushed) -> None:
        """The call ends."""
        self.controller.close()

    def _set(self, state: str) -> None:
        """Tell the controller the agent's new state, when it changed."""
        if state != self._state:
            old, self._state = self._state, state
            self.controller.on_agent_state_changed(old, state)

    async def _speak(self, text: str) -> bool:
        """Say a line and wait until it has played.

        Always returns False: a caller who spoke over the line has already started the
        stages over, so the controller drops what follows.
        """
        if self._model is not None:
            return await self._played_after(speak(self._model, text))
        return await self._played_after(self._worker.queue_frame(TTSSpeakFrame(text)))

    async def _generate(self, instructions: str) -> bool:
        """Have the model word a nudge, and wait until it has played.

        The instructions join the context as a developer message.
        Returns False, as ``_speak`` does.
        """
        if self._model is not None:
            return await self._played_after(ask(self._model, instructions))
        message = ChatCompletionDeveloperMessageParam(role="developer", content=instructions)
        frame = LLMMessagesAppendFrame(messages=[message], run_llm=True)
        return await self._played_after(self._worker.queue_frame(frame))

    async def _played_after(self, start: Awaitable[None]) -> bool:
        """Make the bot speak, and wait until the bot stops speaking."""
        self._played.clear()
        await start
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._played.wait(), timeout=PLAYOUT_TIMEOUT_SECONDS)
        return False


_FRAMES = FrameTable[IdleNudgeObserver](
    [
        (BotStartedSpeakingFrame, IdleNudgeObserver._on_bot_started),
        (BotStoppedSpeakingFrame, IdleNudgeObserver._on_bot_stopped),
        (LLMFullResponseStartFrame, IdleNudgeObserver._on_response_start),
        (LLMFullResponseEndFrame, IdleNudgeObserver._on_response_end),
        (FunctionCallInProgressFrame, IdleNudgeObserver._on_tool_started),
        ((FunctionCallResultFrame, FunctionCallCancelFrame), IdleNudgeObserver._on_tool_ended),
        ((TranscriptionFrame, InterimTranscriptionFrame), IdleNudgeObserver._on_transcript),
        ((EndFrame, CancelFrame), IdleNudgeObserver._on_end),
    ]
)
"""Each frame class and the observer method that handles it, in the order they are tried."""
