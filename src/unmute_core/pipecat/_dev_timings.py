"""The timings of one Pipecat call that no single operation owns: the reply's wait."""

from collections.abc import Mapping

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.user_bot_latency_observer import LatencyBreakdown, UserBotLatencyObserver
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.ai_service import AIService
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from unmute_core.dev import DevRecorder
from unmute_core.model import answered_model, model_name, provider_name
from unmute_core.pipecat._model import ModelCallLLMService
from unmute_core.pipecat._services import service_provider

_NS = 1_000_000_000


def called(service: object, *, answered: bool = False) -> tuple[str | None, str | None]:
    """Return the model and provider a ``ModelCallLLMService``'s model call names now.

    By default the model is the one the next request starts with. With ``answered``, it is
    the one that answered the last request, for a model call that fails over.

    ``hand_off(model=...)`` switches the model call, so the service's settings and its
    metrics name the first model only. Each is None when the model call does not name it,
    and both are None for any other service.
    """
    if not isinstance(service, ModelCallLLMService):
        return None, None
    call = service.model_call
    model = answered_model(call) if answered else model_name(call)
    provider = provider_name(call)
    return (model if model != "unknown" else None), (provider if provider != "unknown" else None)


class Timings:
    """Owns Pipecat's latency observer and writes the measurements that come with a reply."""

    def __init__(self, recorder: DevRecorder) -> None:
        """Start the latency observer and hook its events to the recorder."""
        self._rec = recorder
        self._latency = UserBotLatencyObserver()
        self._latency.add_event_handler("on_latency_measured", self._reply_latency)
        self._latency.add_event_handler("on_first_bot_speech_latency", self._first_speech)
        self._latency.add_event_handler("on_latency_breakdown", self._breakdown)
        self._speech_started: int | None = None

    async def on_push_frame(self, data: FramePushed) -> None:
        """Let the latency observer see a frame."""
        await self._latency.on_push_frame(data)

    def speech_started(self, timestamp: int) -> None:
        """Note when the agent's audio started, in frame time."""
        self._speech_started = timestamp

    def speech_ended(self, timestamp: int) -> None:
        """Record how long the agent spoke, which no service reports, under its reply."""
        if self._speech_started is None:
            return
        seconds, self._speech_started = (timestamp - self._speech_started) / _NS, None
        response = self._rec.response
        self._rec.measure(
            self._rec.new_id("speech-duration"),
            "speech_duration",
            seconds,
            "pipecat-bot-speaking-frames",
            scope="response" if response else "unassigned",
            exchange_id=response,
            reason=None if response else "No reply was open while the agent spoke.",
        )

    async def _reply_latency(
        self, _observer: UserBotLatencyObserver, seconds: float | None
    ) -> None:
        """Record the time from the caller's silence to the agent's first audio."""
        response = self._rec.response
        self._rec.measure(
            f"{response}:reply-latency" if response else self._rec.new_id("reply-latency"),
            "reply_latency",
            seconds,
            "pipecat-user-bot-latency",
            scope="response" if response else "unassigned",
            exchange_id=response,
            reason=None if response else "No reply was open when the agent started to speak.",
        )

    async def _first_speech(self, _observer: UserBotLatencyObserver, seconds: float | None) -> None:
        """Record the time from the client connecting to the agent's first speech."""
        self._rec.measure(
            "call:first-speech", "first_speech", seconds, "pipecat-client-connected", scope="call"
        )

    async def _breakdown(
        self, _observer: UserBotLatencyObserver, breakdown: LatencyBreakdown
    ) -> None:
        """Record the caller's turn decision and where the wait for the reply went.

        The parts name all of the wait, the ones no service measures too, and sum to its
        total. A breakdown names no reply, so the page shows it in arrival order.
        """
        response = self._rec.response
        if breakdown.user_turn_secs is not None:
            self._rec.measure(
                f"{response}:turn-detection" if response else self._rec.new_id("turn-detection"),
                "turn_detection",
                breakdown.user_turn_secs,
                "pipecat-user-turn",
                scope="response" if response else "unassigned",
                exchange_id=response,
                reason=None if response else "No reply was open when the agent started to speak.",
            )
        self._rec.breakdown(
            breakdown.measured_from,
            breakdown.total_secs,
            [
                {
                    "key": part.key,
                    "label": part.label,
                    "owner": part.owner,
                    "owner_kind": str(part.owner_kind),
                    "start_time": part.start_time,
                    "duration_secs": part.duration_secs,
                }
                for part in breakdown.contributions
            ],
        )

    def metrics(self, frame: MetricsFrame, source: FrameProcessor, last: Mapping[str, str]) -> None:
        """Record the first response and the processing time a service reported.

        Args:
            frame: The metrics frame.
            source: The service that pushed it.
            last: The newest operation id of each role (``llm``, ``stt``, ``tts``).
        """
        role = (
            "llm"
            if isinstance(source, LLMService)
            else "tts"
            if isinstance(source, TTSService)
            else "stt"
            if isinstance(source, STTService)
            else None
        )
        for data in frame.data:
            if role is None or data.processor != source.name:
                continue
            if isinstance(data, TextAggregationMetricsData):
                self._text_aggregation(data.value, role, last)
                continue
            if isinstance(data, TTFBMetricsData):
                metric = "first_response"
            elif isinstance(data, ProcessingMetricsData):
                metric = "request_duration"
            else:
                continue
            named_model, provider = called(source, answered=True)
            model = named_model or data.model
            operation = last.get(role)
            exchange = (
                (self._rec.get("operation", operation) or {}).get("exchange_id")
                if operation
                else None
            )
            self._rec.measure(
                f"{operation}:{metric}" if operation else self._rec.new_id(metric),
                metric,
                data.value,
                f"pipecat-{role}-{'ttfb' if metric == 'first_response' else 'processing'}",
                scope="operation" if operation else "unassigned",
                exchange_id=exchange,
                operation_id=operation,
                model=model,
                provider=provider
                or (service_provider(source) if isinstance(source, AIService) else None),
                reason=None
                if operation
                else "The service reported this before it had an operation.",
            )
            if operation and model:
                self._rec.update("operation", operation, model=model)

    def _text_aggregation(self, seconds: float, role: str, last: Mapping[str, str]) -> None:
        """Record the wait from the first token to the first sentence spoken, under its reply."""
        operation = last.get(role)
        response = self._rec.response or (
            (self._rec.get("operation", operation) or {}).get("exchange_id") if operation else None
        )
        self._rec.measure(
            self._rec.new_id("text-aggregation"),
            "text_aggregation",
            seconds,
            "pipecat-text-aggregation",
            scope="response" if response else "unassigned",
            exchange_id=response,
            reason=None if response else "The service reported this before a reply was open.",
        )
