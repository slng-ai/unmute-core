"""Pipeline frames as the spans of ``unmute_core.telemetry``.

The span names and attributes are the contract in docs/core/observe/telemetry.mdx. Pipecat's own tracing
(``PipelineWorker(enable_tracing=True)``) stays off, or the call is traced twice.

Each ``llm`` span carries the request the service got: its system instruction, the context's
messages and the tools offered, then the text and the tool calls the model chose. ``stt``
comes from the STT service's transcripts, ``tts`` from the TTS service's frames.

One turn is one caller message and everything the agent does until the next. The caller
shows up as speech (``UserStartedSpeakingFrame``, ``TranscriptionFrame``) or as typed text
(an ``LLMMessagesAppendFrame`` with a user message, which is how RTVI sends it).

A ``tts`` span ends when its audio has played, at ``BotStoppedSpeakingFrame``, or when the
caller cuts it off. Speech over the agent gives a ``caller.over_agent`` event, as on
LiveKit: an interruption when the caller's turn starts while the agent speaks, or a kept
answer when their words wait for the agent's line to end.
"""

import json
import logging
import time
from typing import Any, Literal, cast

from opentelemetry.trace import Span
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pydantic import BaseModel, ConfigDict

from unmute_core.pipecat._frames import FrameTable
from unmute_core.pipecat._model import ModelCallLLMService
from unmute_core.pipecat._realtime import is_realtime
from unmute_core.pipecat._services import service_provider, service_setting, typed_text
from unmute_core.pipecat._tasks import RequestSpans, UnmuteFlows
from unmute_core.pipecat._trace_requests import RequestTracker
from unmute_core.pipecat._trace_speech import SpeechTracker
from unmute_core.tasks import Opening
from unmute_core.telemetry import Telemetry, TurnTracker

_NS = 1_000_000_000

logger = logging.getLogger(__name__)


def _language(language: object) -> str | None:
    """Return a language as its code, like ``en``, from a Pipecat ``Language`` or a string."""
    return str(getattr(language, "value", language)) if language else None


class _Latency(BaseModel):
    """The waits of one turn, in seconds. Each is None until it is first seen."""

    model_config = ConfigDict(extra="forbid")

    e2e: float | None = None
    eou: float | None = None
    stt_final: float | None = None
    llm_ttft: float | None = None
    tts_ttfb: float | None = None

    def note(
        self, name: Literal["e2e", "eou", "stt_final", "llm_ttft", "tts_ttfb"], value: float
    ) -> None:
        """Keep a wait the first time it is seen in the turn."""
        if getattr(self, name) is None:
            setattr(self, name, value)


class TelemetryObserver(BaseObserver):
    """Records each turn, model request and tool call of one call.

    A frame passes several processors. The worker shows this observer its first push only.
    """

    def __init__(
        self,
        telemetry: Telemetry,
        *,
        model: str,
        provider: str,
        flows: UnmuteFlows | None = None,
    ) -> None:
        """Watch the pipeline for one call.

        Args:
            telemetry: The call's telemetry, with its call span started.
            model: The model name the ``llm`` spans carry. After a handoff that gave the
                next agent its own model call, its spans name that model call instead.
            provider: Who serves the model, like ``openai`` or ``slng``.
            flows: The call's tasks, which tell each request's prompt site.
        """
        super().__init__(observe_every_push=False)
        self._telemetry = telemetry
        self._model = model
        self._provider = provider
        self._flows = flows
        telemetry.describe_call(framework="pipecat")
        self._turn = TurnTracker(telemetry)
        self._last_turn: Span | None = None
        self._pipeline_ended = False
        self._user: list[str] = []
        self._lines: list[str] = []
        self._latency = _Latency()
        self._interrupted = False
        self._speaking = False
        self._stopped_at: int | None = None
        self._requests = RequestTracker(
            telemetry,
            model=model,
            provider=provider,
            flows=flows,
            turn_span=self._turn_span,
            said=self._said,
            side_turn=self._side_turn,
        )
        self._tools: dict[str, Span] = {}
        self._running: dict[str, Span] = {}
        self._speech = SpeechTracker(telemetry, self._agent_span)
        self._speaking_since: int | None = None
        self._playing = False
        self._caller_turn = False
        self._kept = False
        self._metrics_checked = False
        if flows is not None:
            flows.request_spans = RequestSpans(self._requests.start_side, self._requests.end_side)
            if isinstance(service := flows.llm, ModelCallLLMService):
                service.trace_headers = self.trace_headers

    def trace_headers(self) -> dict[str, str]:
        """Return the ``traceparent`` of the open turn, or nothing between turns.

        A request sent with it makes the router's spans children of the turn. The request's
        own ``llm`` span may open after the request leaves, so it cannot be the parent.
        """
        return self._telemetry.headers(self._turn.span) if self._turn.span is not None else {}

    async def on_push_frame(self, data: FramePushed) -> None:
        """Turn one frame into span starts, ends and attributes."""
        if not self._metrics_checked and isinstance(data.source, LLMService):
            self._check_metrics(data.source)
        if self._running:
            self._end_tasks()
        if self._flows is not None and self._flows.openings:
            self._openings(self._flows.openings)
        if (
            data.frame.broadcast_sibling_id is not None
            and data.direction == FrameDirection.UPSTREAM
        ):
            return
        if (handler := _FRAMES.handler(data.frame)) is not None:
            handler(self, data)

    def _check_metrics(self, llm: LLMService[Any]) -> None:
        """Warn once when the LLM service reports no metrics, so spans miss tokens and TTFB."""
        self._metrics_checked = True
        if not (llm.metrics_enabled and llm.usage_metrics_enabled):
            logger.warning(
                "the llm spans get no tokens and no time to first token: turn on "
                "PipelineParams(enable_metrics=True, enable_usage_metrics=True)"
            )

    def _on_user_started(self, data: FramePushed) -> None:
        """Start the caller's turn, cutting off the agent's speech when it plays."""
        if self._playing:
            self._over_agent(interruption=True)
        self._caller_turn = True
        self._speech.end_all(interrupted=True)
        self._caller(interrupting=self._speaking)
        if self._speaking_since is None:
            self._speaking_since = time.time_ns()

    def _on_user_stopped(self, data: FramePushed) -> None:
        """Note when the caller's turn ended, for the end-of-utterance wait."""
        self._caller_turn = False
        self._since_stop("eou", data.timestamp)

    def _on_vad_started(self, data: FramePushed) -> None:
        """Forget the last stop, because the caller speaks again."""
        self._stopped_at = None

    def _on_vad_stopped(self, data: FramePushed) -> None:
        """Remember when the caller's voice stopped, as the start of each wait."""
        self._stopped_at = data.timestamp

    def _on_typed(self, data: FramePushed) -> None:
        """Add the text a client typed to the caller's turn."""
        if typed := typed_text(cast(LLMMessagesAppendFrame, data.frame)):
            self._caller()
            self._user.extend(typed)

    def _on_transcript(self, data: FramePushed) -> None:
        """Add a final transcript to the caller's turn, and record its ``stt`` span."""
        frame = cast(TranscriptionFrame, data.frame)
        self._kept = self._kept or (self._playing and not self._caller_turn)
        self._caller()
        self._user.append(frame.text)
        self._since_stop("stt_final", data.timestamp)
        if isinstance(data.source, STTService):
            self._stt(frame, data.source)

    def _on_bot_started(self, data: FramePushed) -> None:
        """Note that the agent's audio is playing."""
        self._speaking = self._playing = True
        self._since_stop("e2e", data.timestamp)

    def _on_bot_stopped(self, data: FramePushed) -> None:
        """End the ``tts`` spans whose audio has played, and note a kept answer."""
        self._speaking = self._playing = False
        self._speech.end_played()
        if self._kept:
            self._over_agent(interruption=False)

    def _on_context(self, data: FramePushed) -> None:
        """Keep the context the next request answers."""
        self._requests.hold_context(cast(LLMContextFrame, data.frame))

    def _on_response_start(self, data: FramePushed) -> None:
        """Open an ``llm`` span for a model reply, unless it continues the open one."""
        if isinstance(source := data.source, LLMService) and not self._requests.same_reply(source):
            self._requests.start(source)

    def _on_calls_started(self, data: FramePushed) -> None:
        """Add the tool calls the model chose to its request."""
        source = data.source
        if not self._requests.is_open and isinstance(source, LLMService) and is_realtime(source):
            self._requests.start(source)
        self._requests.add_calls(cast(FunctionCallsStartedFrame, data.frame).function_calls)

    def _on_speak(self, data: FramePushed) -> None:
        """Add a line the agent was told to say to the open turn."""
        self._said(cast(TTSSpeakFrame, data.frame).text)

    def _on_speech_frame(self, data: FramePushed) -> None:
        """Pass a TTS frame to the ``tts`` spans. After the pipeline ended, none waits to play."""
        self._speech.frame(data)
        if self._pipeline_ended:
            self._speech.end_played()

    def _on_text(self, data: FramePushed) -> None:
        """Add a piece of the model's reply to its request."""
        if isinstance(data.source, LLMService):
            self._requests.add_text(cast(LLMTextFrame, data.frame).text)

    def _on_interruption(self, data: FramePushed) -> None:
        """Note that the open request was cut off."""
        self._requests.mark_cut()

    def _on_response_end(self, data: FramePushed) -> None:
        """Close the ``llm`` span when its service ends the reply."""
        if isinstance(data.source, LLMService):
            self._requests.end_reply()

    def _on_metrics(self, data: FramePushed) -> None:
        """Read token use and time to first byte from a service's metrics."""
        self._requests.metrics(cast(MetricsFrame, data.frame), data.source, self._latency.note)

    def _on_tool_started(self, data: FramePushed) -> None:
        """Open a tool call's span."""
        frame = cast(FunctionCallInProgressFrame, data.frame)
        self._tools[frame.tool_call_id] = self._telemetry.start_tool_call(
            name=frame.function_name,
            call_id=frame.tool_call_id,
            arguments=json.dumps(frame.arguments, ensure_ascii=False),
            parent=self._agent_span(),
        )

    def _on_tool_result(self, data: FramePushed) -> None:
        """End a tool call's span with its result."""
        frame = cast(FunctionCallResultFrame, data.frame)
        if (span := self._tools.pop(frame.tool_call_id, None)) is not None:
            self._tool_result(span, frame)

    def _on_tool_cancelled(self, data: FramePushed) -> None:
        """End a tool call's span as cancelled."""
        frame = cast(FunctionCallCancelFrame, data.frame)
        if (span := self._tools.pop(frame.tool_call_id, None)) is not None:
            self._telemetry.end_tool_call(span, error="cancelled")

    def _on_error(self, data: FramePushed) -> None:
        """Give the open request the type of the error its own service reported."""
        self._requests.note_error(cast(ErrorFrame, data.frame), data.source)

    def _on_end(self, data: FramePushed) -> None:
        """End everything still open when the pipeline ends or is cancelled."""
        self._pipeline_ended = True
        self._requests.end(interrupted=True)
        for span in self._running.values():
            self._telemetry.end_tool_call(span, error="cancelled")
        self._running.clear()
        self._speech.end_all(interrupted=isinstance(data.frame, CancelFrame))
        self._end_turn()

    def _tool_result(self, span: Span, frame: FunctionCallResultFrame) -> None:
        """End a tool call's span with what the tool returned.

        A task call's first result only says the task is running, so its span stays open
        until the task returns. A step's closing tool shows its real result, not the status.
        """
        flows = self._flows
        if flows is not None and frame.tool_call_id in flows.task_results:
            self._running[frame.tool_call_id] = span
            return
        result = frame.result
        if flows is not None and frame.function_name in flows.closing_results:
            result = flows.closing_results.pop(frame.function_name)
        text = json.dumps(result, ensure_ascii=False, default=str)
        self._telemetry.end_tool_call(span, result=text, error=frame.error)

    def _openings(self, openings: list[Opening]) -> None:
        """Record each tool a step ran in code as it opened, and empty the list."""
        for opening in openings:
            span = self._telemetry.start_tool_call(
                name=opening.tool,
                call_id=opening.call_id,
                arguments=opening.arguments,
                parent=self._agent_span(),
            )
            result = json.dumps(opening.result, ensure_ascii=False, default=str)
            self._telemetry.end_tool_call(span, result=result)
        openings.clear()

    def _end_tasks(self) -> None:
        """End the span of each task call whose task has returned, with the main agent's result."""
        assert self._flows is not None
        results = self._flows.task_results
        for call_id in [call_id for call_id in self._running if results.get(call_id) is not None]:
            self._telemetry.end_tool_call(self._running.pop(call_id), result=results.pop(call_id))

    def _over_agent(self, *, interruption: bool) -> None:
        """Record what the caller's speech over the agent did: cut it off, or wait for it."""
        self._kept = False
        self._telemetry.event(
            "caller.over_agent", {"interruption": interruption}, parent=self._turn.span
        )

    def _since_stop(self, name: Literal["e2e", "eou", "stt_final"], timestamp: int) -> None:
        """Note the seconds from the caller's last stop to now, once per turn."""
        if self._stopped_at is not None:
            self._latency.note(name, (timestamp - self._stopped_at) / _NS)

    def _caller(self, *, interrupting: bool = False) -> None:
        """Start the next turn on a sign of the caller, unless the open turn is still theirs.

        A turn is the caller's until the agent does something in it. More speech or text
        before that, like a pause mid-sentence, joins the open turn.

        Args:
            interrupting: True when the caller speaks over the agent, which cuts that turn off.
        """
        if self._turn.caller_open:
            return
        self._interrupted = self._interrupted or interrupting
        self._end_turn()
        self._turn.start()

    def _turn_span(self) -> Span:
        """Return the open turn, opening one when the agent speaks first."""
        return self._turn.open(answering=False)

    def _agent_span(self) -> Span:
        """Return the open turn for something the agent does, which ends the caller's part.

        Once the pipeline ended it is the last turn: a line that plays as the call ends,
        like the apology for a failure, reaches the TTS after the end frame passed by.
        """
        if self._pipeline_ended and self._last_turn is not None:
            return self._last_turn
        return self._turn.open()

    def _side_turn(self) -> Span:
        """Return the turn a side request belongs to: the last one, once the pipeline ended."""
        if self._pipeline_ended and self._last_turn is not None:
            return self._last_turn
        return self._turn_span()

    def _said(self, text: str) -> None:
        """Add one line the agent said to the open turn."""
        if text.strip():
            self._agent_span()
            self._lines.append(text.strip())

    def _end_turn(self) -> None:
        """Close the open turn with what was said and how fast."""
        if self._turn.span is None:
            return
        self._last_turn = self._turn.span
        self._turn.end(
            user_text=" ".join(self._user) or None,
            agent_lines=self._lines,
            interrupted=self._interrupted,
            e2e=self._latency.e2e,
            eou=self._latency.eou,
            stt_final=self._latency.stt_final,
            llm_ttft=self._latency.llm_ttft,
            tts_ttfb=self._latency.tts_ttfb,
        )
        self._user, self._lines, self._latency = [], [], _Latency()
        self._interrupted = self._speaking = False
        self._speaking_since = None

    def _stt(self, frame: TranscriptionFrame, service: STTService) -> None:
        """Record one transcript as an ``stt`` span, from when the caller started speaking."""
        self._telemetry.stt(
            text=frame.text,
            model=service_setting(service, "model") or "unknown",
            provider=service_provider(service),
            start_time=self._speaking_since,
            language=_language(frame.language or getattr(service.settings, "language", None)),
            ttfb=self._latency.stt_final,
            parent=self._turn_span(),
        )


_FRAMES = FrameTable[TelemetryObserver](
    [
        (UserStartedSpeakingFrame, TelemetryObserver._on_user_started),
        (UserStoppedSpeakingFrame, TelemetryObserver._on_user_stopped),
        (VADUserStartedSpeakingFrame, TelemetryObserver._on_vad_started),
        (VADUserStoppedSpeakingFrame, TelemetryObserver._on_vad_stopped),
        (LLMMessagesAppendFrame, TelemetryObserver._on_typed),
        (TranscriptionFrame, TelemetryObserver._on_transcript),
        (BotStartedSpeakingFrame, TelemetryObserver._on_bot_started),
        (BotStoppedSpeakingFrame, TelemetryObserver._on_bot_stopped),
        (LLMContextFrame, TelemetryObserver._on_context),
        (LLMFullResponseStartFrame, TelemetryObserver._on_response_start),
        (FunctionCallsStartedFrame, TelemetryObserver._on_calls_started),
        (TTSSpeakFrame, TelemetryObserver._on_speak),
        (
            (TTSStartedFrame, TTSTextFrame, TTSAudioRawFrame, TTSStoppedFrame),
            TelemetryObserver._on_speech_frame,
        ),
        (LLMTextFrame, TelemetryObserver._on_text),
        (InterruptionFrame, TelemetryObserver._on_interruption),
        (LLMFullResponseEndFrame, TelemetryObserver._on_response_end),
        (ErrorFrame, TelemetryObserver._on_error),
        (MetricsFrame, TelemetryObserver._on_metrics),
        (FunctionCallInProgressFrame, TelemetryObserver._on_tool_started),
        (FunctionCallResultFrame, TelemetryObserver._on_tool_result),
        (FunctionCallCancelFrame, TelemetryObserver._on_tool_cancelled),
        ((EndFrame, CancelFrame), TelemetryObserver._on_end),
    ]
)
"""Each frame class and the observer method that handles it, in the order they are tried."""
