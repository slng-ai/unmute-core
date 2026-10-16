"""The ``tts`` spans of one Pipecat call: each piece of speech from first frame to played."""

from collections.abc import Callable

from opentelemetry.trace import Span
from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.services.tts_service import TTSService

from unmute_core.pipecat._services import service_provider, service_setting
from unmute_core.telemetry import Telemetry

_NS = 1_000_000_000


class Speech:
    """One piece of speech the TTS service is making: its span, text and first audio.

    ``made`` turns True once TTS has made all of it. It has played at the next
    ``BotStoppedSpeakingFrame``.
    """

    def __init__(self, span: Span, started: int) -> None:
        """Keep the span and when TTS started, in frame time."""
        self.span = span
        self.started = started
        self.text: list[str] = []
        self.ttfb: float | None = None
        self.made = False

    def add(self, frame: TTSTextFrame) -> None:
        """Add the text of one frame, with a space when the frame carries none."""
        self.text.append(
            frame.text if frame.includes_inter_frame_spaces else f" {frame.text.strip()}"
        )

    def spoken(self) -> str:
        """Return the whole text."""
        return "".join(self.text).strip()


class SpeechTracker:
    """Owns the open ``tts`` spans of a call, one per TTS context."""

    def __init__(self, telemetry: Telemetry, agent_span: Callable[[], Span]) -> None:
        """Keep the telemetry and how to find the turn a speech belongs to.

        Args:
            telemetry: The call's telemetry.
            agent_span: Returns the open turn for something the agent does.
        """
        self._telemetry = telemetry
        self._agent_span = agent_span
        self._speech: dict[str, Speech] = {}

    def frame(self, data: FramePushed) -> None:
        """Open, fill and end ``tts`` spans, one per TTS context."""
        frame = data.frame
        key = getattr(frame, "context_id", None) or ""
        if isinstance(frame, TTSStartedFrame) and isinstance(data.source, TTSService):
            span = self._telemetry.start_tts(
                model=service_setting(data.source, "model") or "unknown",
                provider=service_provider(data.source),
                voice=service_setting(data.source, "voice"),
                parent=self._agent_span(),
            )
            self._speech[key] = Speech(span, data.timestamp)
        elif (speech := self._speech.get(key)) is None:
            return
        elif isinstance(frame, TTSTextFrame):
            speech.add(frame)
        elif isinstance(frame, TTSAudioRawFrame) and speech.ttfb is None:
            speech.ttfb = (data.timestamp - speech.started) / _NS
        elif isinstance(frame, TTSStoppedFrame):
            speech.made = True
            if speech.ttfb is None:
                del self._speech[key]
                self._telemetry.end_tts(speech.span, text=speech.spoken())

    def end_played(self) -> None:
        """End each ``tts`` span whose audio was all made, now that the bot has stopped."""
        for key in [key for key, speech in self._speech.items() if speech.made]:
            speech = self._speech.pop(key)
            self._telemetry.end_tts(speech.span, text=speech.spoken(), ttfb=speech.ttfb)

    def end_all(self, *, interrupted: bool) -> None:
        """End every ``tts`` span still open."""
        for speech in self._speech.values():
            self._telemetry.end_tts(
                speech.span, text=speech.spoken(), ttfb=speech.ttfb, interrupted=interrupted
            )
        self._speech.clear()
