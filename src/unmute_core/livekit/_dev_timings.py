"""The timings of one LiveKit call that come from its clock and its chat messages."""

import time
from typing import Any

from unmute_core.dev import DevRecorder


class Timings:
    """Owns the call's first-speech clock and writes the timings LiveKit puts on messages."""

    def __init__(self, recorder: DevRecorder) -> None:
        """Start the clock at now, until the session attaches or the agent starts."""
        self._rec = recorder
        self.attached_at = time.time()
        self._started_at: float | None = None
        self._first_speech_at: float | None = None

    def attached(self) -> None:
        """Note that the observer attached to a session now."""
        self.attached_at = time.time()

    def started(self, at: float) -> None:
        """Note when the agent started, once. The first speech counts from it."""
        if self._started_at is None:
            self._started_at = at

    def first_speech(self, at: float, source: str) -> None:
        """Record the call's first speech, keeping the earliest start seen.

        Args:
            at: When the speech started, as LiveKit's clock reads.
            source: Where the time came from.
        """
        if self._rec.closed or (self._first_speech_at is not None and at >= self._first_speech_at):
            return
        self._first_speech_at = at
        self._rec.measure(
            "call:first-speech",
            "first_speech",
            at - (self._started_at or self.attached_at),
            source,
            scope="call",
        )

    def turn(self, response: str, metrics: dict[str, Any]) -> None:
        """Record the timings LiveKit puts on the caller's message, under its reply."""
        for key, metric in (
            ("end_of_turn_delay", "turn_detection"),
            ("transcription_delay", "transcription_delay"),
        ):
            if key in metrics:
                self._rec.measure(
                    f"{response}:{metric}",
                    metric,
                    metrics[key],
                    "livekit-chat-message",
                    scope="response",
                    exchange_id=response,
                )

    def reply(self, response: str, message: str, metrics: dict[str, Any]) -> None:
        """Record the timings LiveKit puts on one reply message.

        Args:
            response: The response the reply belongs to.
            message: The reply message's id, which names its ``first_response`` timings.
            metrics: The message's metrics.
        """
        rec = self._rec
        source = "livekit-chat-message"
        latency = metrics.get("e2e_latency")
        if latency is not None and rec.get("measurement", f"{response}:reply-latency") is None:
            rec.measure(
                f"{response}:reply-latency",
                "reply_latency",
                latency,
                source,
                scope="response",
                exchange_id=response,
            )
        for field, reason in (
            ("llm_node_ttft", "The response's model node to its first output."),
            ("tts_node_ttfb", "The response's synthesis node to its first audio."),
        ):
            if field in metrics:
                rec.measure(
                    f"{response}:{message}:{field}",
                    "first_response",
                    metrics[field],
                    f"{source}.{field}",
                    scope="response",
                    exchange_id=response,
                    reason=reason,
                )
        started, stopped = metrics.get("started_speaking_at"), metrics.get("stopped_speaking_at")
        first, sentence = metrics.get("llm_node_ttft"), metrics.get("llm_node_ttfs")
        for metric, value in (
            ("speech_duration", stopped - started if started and stopped else None),
            ("playback_delay", metrics.get("playback_latency")),
            ("text_aggregation", sentence - first if sentence is not None and first else None),
        ):
            if value is not None:
                rec.measure(
                    rec.new_id(metric.replace("_", "-")),
                    metric,
                    value,
                    source,
                    scope="response",
                    exchange_id=response,
                )
