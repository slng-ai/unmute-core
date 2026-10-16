"""Pipeline frames as the dev records of ``unmute_core.dev``.

The observer sees each frame once, as the worker shows it, and keeps the call's exchanges.
An input exchange is the caller speaking, or typing. A response exchange is everything the
agent does until the caller speaks again: its model requests, its tool calls, its speech.
Both hold the text of what was said, as rows that are first provisional, then final.

Operations are one model request (``llm``), one transcript (``stt``), one piece of speech
(``tts``) and one tool call (``tool``). A task's handoff comes from ``DevRecorder.task_event``.

A response is linked to the input it answers when it opens right after one. Pipecat does not
say which request answers which input, so this is the order of the frames, not a proof.
"""

import logging
from collections.abc import Collection
from typing import Literal, cast

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_service import AIService
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.dev import DevRecorder
from unmute_core.pipecat._dev_timings import Timings, called
from unmute_core.pipecat._dev_tools import ToolRuns
from unmute_core.pipecat._frames import FrameTable
from unmute_core.pipecat._services import service_provider, service_setting, typed_text

logger = logging.getLogger(__name__)

_NS = 1_000_000_000


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class _Started:
    """An operation that is running, and the frame time it started."""

    operation: str
    timestamp: int
    service: object = None


class DevObserver(BaseObserver):
    """Sends the dev records of one call from the pipeline's frames.

    Give it to the worker: ``PipelineWorker(pipeline, observers=[observer])``. It sends the
    exchanges, the caller's text and the agent's, the operations (``llm``, ``stt``, ``tts``,
    ``tool``) and these measurements: ``first_response`` of each model request, speech and
    transcript, ``request_duration`` of each model request, ``tool_duration`` of each tool
    call, ``reply_latency``, ``turn_detection``, ``speech_duration`` and
    ``text_aggregation`` of each reply, and ``first_speech`` of the call. Each reply's wait
    also gives a ``breakdown`` record, part by part. Its end ends the call's records.

    A frame passes several processors. The worker shows this observer its first push only.
    """

    def __init__(
        self,
        recorder: DevRecorder,
        *,
        handoffs: Collection[str] = (),
        controls: Collection[str] = (),
    ) -> None:
        """Watch the pipeline for one call.

        Args:
            recorder: The call's recorder. ``recorder.task_event`` and ``recorder.watch_state``
                are for its tasks and its state.
            handoffs: The tools that start a task or a group, like ``tasks.delegate_tools()``
                names. They get no ``tool`` operation, because ``recorder.task_event`` draws
                the run as a ``handoff`` operation.
            controls: The tools that hand the call over without a result: agent handoff tools
                and human transfer tools. They get a ``handoff`` operation and no
                ``tool_duration``.
        """
        super().__init__(observe_every_push=False)
        self._rec = recorder
        self._input: str | None = None
        self._unlinked: str | None = None
        self._stt: str | None = None
        self._speaking = False
        self._spoke = False
        self._llm: _Started | None = None
        self._answer: str | None = None
        self._tts: dict[str, str] = {}
        self._tools = ToolRuns(recorder, handoffs=handoffs, controls=controls)
        self._last: dict[str, str] = {}
        self._timings = Timings(recorder)

    def finish(self, *, error: bool = False, reason: str | None = None) -> None:
        """End the call's records. The observer does it at the end frame. Call it when none comes.

        Args:
            error: True when the call ended in an error.
            reason: What ended it.
        """
        self._rec.end_call(error=error, reason=reason)

    async def on_push_frame(self, data: FramePushed) -> None:
        """Turn one frame into records."""
        await self._timings.on_push_frame(data)
        if self._rec.closed or (
            data.frame.broadcast_sibling_id is not None
            and data.direction == FrameDirection.UPSTREAM
        ):
            return
        if (handler := _FRAMES.handler(data.frame)) is not None:
            handler(self, data)

    def _on_user_started(self, data: FramePushed) -> None:
        """Open an input exchange for the caller's turn."""
        self._caller_starts()

    def _on_typed(self, data: FramePushed) -> None:
        """Record each message a client typed as the caller's final text."""
        for text in typed_text(cast(LLMMessagesAppendFrame, data.frame)):
            self._caller_starts()
            self._user_text(text, final=True, spacing=True, source=data.source)

    def _on_transcript(self, data: FramePushed) -> None:
        """Record a transcript, provisional until it is final."""
        frame = cast(InterimTranscriptionFrame | TranscriptionFrame, data.frame)
        self._user_text(
            frame.text,
            final=isinstance(frame, TranscriptionFrame),
            spacing=frame.includes_inter_frame_spaces,
            source=data.source,
        )

    def _on_context(self, data: FramePushed) -> None:
        """End the open input, because the context holding its words reached the model."""
        self._end_input()

    def _on_response_start(self, data: FramePushed) -> None:
        """Open an ``llm`` operation when a model's reply starts."""
        if isinstance(data.source, LLMService):
            self._start_llm(data.source, data.timestamp)

    def _on_text(self, data: FramePushed) -> None:
        """Add the model's words to the answer."""
        if isinstance(data.source, LLMService):
            self._llm_text(cast(LLMTextFrame, data.frame))

    def _on_response_end(self, data: FramePushed) -> None:
        """Settle the ``llm`` operation when its reply ends."""
        if isinstance(data.source, LLMService):
            self._end_llm("ended", data.timestamp)

    def _on_speak(self, data: FramePushed) -> None:
        """Record a line the agent was told to say."""
        self._direct_text(cast(TTSSpeakFrame, data.frame).text)

    def _on_tts_started(self, data: FramePushed) -> None:
        """Open a ``tts`` operation when a TTS service starts a piece of speech."""
        if isinstance(data.source, TTSService):
            self._start_tts(cast(TTSStartedFrame, data.frame).context_id or "", data.source)

    def _on_tts_stopped(self, data: FramePushed) -> None:
        """End the ``tts`` operation of a piece of speech."""
        context_id = cast(TTSStoppedFrame, data.frame).context_id
        if (operation := self._tts.pop(context_id, None)) is not None:
            self._rec.update("operation", operation, state="ended")

    def _on_tool_started(self, data: FramePushed) -> None:
        """Open a ``tool`` operation."""
        frame = cast(FunctionCallInProgressFrame, data.frame)
        self._tools.start(
            frame,
            data.timestamp,
            exchange_id=self._rec.response or self._input,
            parent=self._llm.operation if self._llm is not None else self._last.get("llm"),
        )

    def _on_tool_result(self, data: FramePushed) -> None:
        """End a ``tool`` operation with its result."""
        self._tools.result(cast(FunctionCallResultFrame, data.frame), data.timestamp)

    def _on_tool_cancelled(self, data: FramePushed) -> None:
        """End a ``tool`` operation that was cancelled or ran past its deadline."""
        self._tools.cancelled(cast(FunctionCallCancelFrame, data.frame), data.timestamp)

    def _on_bot_started(self, data: FramePushed) -> None:
        """Note that the agent's audio is playing."""
        self._speaking = self._spoke = True
        self._timings.speech_started(data.timestamp)
        self._playback("speaking")

    def _on_bot_stopped(self, data: FramePushed) -> None:
        """Time the agent's speech and settle the open response."""
        self._speaking = False
        self._timings.speech_ended(data.timestamp)
        self._bot_stopped()

    def _on_interruption(self, data: FramePushed) -> None:
        """Cancel what the caller cut off."""
        self._interrupted(data.timestamp)

    def _on_metrics(self, data: FramePushed) -> None:
        """Record what a service measured."""
        self._timings.metrics(cast(MetricsFrame, data.frame), data.source, self._last)

    def _on_error(self, data: FramePushed) -> None:
        """Mark the call when a service reports a fatal error."""
        self._error(cast(ErrorFrame, data.frame), data.source)

    def _on_end(self, data: FramePushed) -> None:
        """End the call's records at the end frame, cancelling what a cancel cut off."""
        if isinstance(data.frame, CancelFrame):
            self._interrupted(data.timestamp)
        self._rec.end_call()

    def _caller_starts(self) -> None:
        """Open an input exchange for the caller's turn, unless one is open and unanswered.

        The turn can start after its first words: Pipecat's turn strategy pushes
        ``UserStartedSpeakingFrame`` once the transcript has begun. Those words stay in it.
        An input a reply already answers, still open, is incomplete.
        """
        if self._input and self._input == self._unlinked:
            return
        if self._input:
            self._rec.update("exchange", self._input, state="incomplete")
        self._settle_response()
        self._input = self._unlinked = self._rec.open_input()

    def _end_input(self) -> None:
        """End the open input exchange, because its words went to the model."""
        if self._input:
            self._rec.update("exchange", self._input, state="ended")
            self._input = None

    def _user_text(self, text: str, *, final: bool, spacing: bool, source: FrameProcessor) -> None:
        """Record what the caller said or typed as a text row, provisional until final.

        Args:
            text: The words so far.
            final: True for the final transcript of this segment.
            spacing: True when the text carries its own spaces.
            source: The processor that pushed it, which names the ``stt`` operation.
        """
        if self._input is None:
            self._caller_starts()
        assert self._input is not None
        self._rec.user_text(self._input, text, final=final, spacing=spacing)
        if isinstance(source, STTService):
            self._stt_operation(source, final)

    def _stt_operation(self, service: STTService, final: bool) -> None:
        """Keep an ``stt`` operation running from the first words of a transcript to its end."""
        if self._stt is None:
            self._stt = self._last["stt"] = self._open_operation("stt", service)
        if final:
            self._rec.update("operation", self._stt, state="ended")
            self._stt = None

    def _open_operation(
        self,
        kind: Literal["llm", "stt", "tts"],
        service: AIService,
    ) -> str:
        """Open an operation of a service, in the open response if any, and return its id."""
        named, provider = called(service)
        scope = self._rec.response or self._input
        return self._rec.operation(
            kind,
            service.name,
            exchange_id=scope if scope and (kind != "stt" or self._rec.response is None) else None,
            model=named or service_setting(service, "model"),
            provider=provider or service_provider(service),
            provider_first=True,
        )

    def _response(self) -> str:
        """Return the open response exchange, or open one linked to the input it answers."""
        if self._rec.response is not None:
            return self._rec.response
        identity = self._rec.open_response(self._unlinked)
        self._unlinked = None
        self._spoke = False
        return identity

    def _playback(self, state: str) -> None:
        """Say how far the agent's speech of the open response has got."""
        if self._rec.response is not None:
            self._rec.update("exchange", self._rec.response, playback=state)

    def _busy(self) -> bool:
        """Return True while the agent is speaking or a model request or speech is running."""
        return self._speaking or self._llm is not None or bool(self._tts)

    def _settle_response(self) -> None:
        """Close the open response when the caller speaks again: ended, or cut off."""
        if (response := self._rec.response) is None:
            return
        cut = self._busy()
        self._rec.settle_exchange(response, cut=cut, spoke=self._spoke or cut)
        self._rec.close_response()

    def _bot_stopped(self) -> None:
        """End the open response when the agent has said its piece and nothing else runs."""
        if self._rec.response is not None:
            self._playback("ended")
            if not self._busy() and not self._tools.running:
                self._rec.update("exchange", self._rec.response, state="ended")
                self._rec.close_response()

    def _start_llm(self, service: LLMService, timestamp: int) -> None:
        """Open an ``llm`` operation, with its two timings pending."""
        self._end_llm("cancelled", timestamp)
        response = self._response()
        operation = self._open_operation("llm", service)
        self._llm, self._answer = _Started(operation, timestamp, service), None
        self._last["llm"] = operation
        self._rec.pending_request(operation, response, "pipecat-llm")

    def _llm_text(self, frame: LLMTextFrame) -> None:
        """Add the model's words to the answer's text row."""
        if self._llm is None or not frame.text:
            return
        if self._answer is None:
            self._answer = self._rec.new_id("answer-text")
        self._rec.append_text(
            self._answer,
            self._response(),
            frame.text,
            spacing=frame.includes_inter_frame_spaces,
        )

    def _end_llm(self, state: str, timestamp: int) -> None:
        """Settle the running model request, its text and the timings it left pending."""
        if self._llm is None:
            return
        running, self._llm = self._llm, None
        operation, started = running.operation, running.timestamp
        answer, self._answer = self._answer, None
        if (answered := called(running.service, answered=True)[0]) is not None:
            self._rec.update("operation", operation, model=answered)
        self._rec.settle_request(
            operation,
            state,
            timings={"request_duration": (timestamp - started) / _NS},
            source="pipecat-frames",
            unavailable=f"The request {state} before this measurement arrived.",
            answer=answer,
        )

    def _direct_text(self, text: str) -> None:
        """Record a line the agent was told to say, like a greeting, as final text."""
        if text.strip():
            self._rec.direct_text(self._response(), text)

    def _start_tts(self, context_id: str, service: TTSService) -> None:
        """Open a ``tts`` operation for one piece of speech."""
        self._response()
        self._tts[context_id] = self._last["tts"] = self._open_operation("tts", service)

    def _interrupted(self, timestamp: int) -> None:
        """Cancel what the caller cut off: the model request, the speech, the tool calls."""
        self._end_llm("cancelled", timestamp)
        for operation in self._tts.values():
            self._rec.update("operation", operation, state="cancelled")
        self._tts.clear()
        self._tools.cancel_all(timestamp)

    def _error(self, frame: ErrorFrame, source: FrameProcessor) -> None:
        """Mark the call as an error when a service reports a fatal one, with its name only."""
        if frame.fatal:
            self._rec.mark_error(f"{source.name} reported a fatal error.")


_FRAMES = FrameTable[DevObserver](
    [
        (UserStartedSpeakingFrame, DevObserver._on_user_started),
        (LLMMessagesAppendFrame, DevObserver._on_typed),
        ((InterimTranscriptionFrame, TranscriptionFrame), DevObserver._on_transcript),
        (LLMContextFrame, DevObserver._on_context),
        (LLMFullResponseStartFrame, DevObserver._on_response_start),
        (LLMTextFrame, DevObserver._on_text),
        (LLMFullResponseEndFrame, DevObserver._on_response_end),
        (TTSSpeakFrame, DevObserver._on_speak),
        (TTSStartedFrame, DevObserver._on_tts_started),
        (TTSStoppedFrame, DevObserver._on_tts_stopped),
        (FunctionCallInProgressFrame, DevObserver._on_tool_started),
        (FunctionCallResultFrame, DevObserver._on_tool_result),
        (FunctionCallCancelFrame, DevObserver._on_tool_cancelled),
        (BotStartedSpeakingFrame, DevObserver._on_bot_started),
        (BotStoppedSpeakingFrame, DevObserver._on_bot_stopped),
        (InterruptionFrame, DevObserver._on_interruption),
        (MetricsFrame, DevObserver._on_metrics),
        (ErrorFrame, DevObserver._on_error),
        ((EndFrame, CancelFrame), DevObserver._on_end),
    ]
)
"""Each frame class and the observer method that handles it, in the order they are tried."""
