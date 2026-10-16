"""What speech over the agent does on Pipecat: cut the agent off, or wait for it.

LiveKit's adaptive detector judges speech over the agent by its sound. Pipecat has no such
detector, so ``keep`` counts words: a short answer like "Yes, please." waits for the
agent's line to end and then answers it, while a longer one cuts the agent off. A
three-word answer like "Yes, that's right" therefore cuts the line, where LiveKit would
keep it.
"""

import asyncio
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_mute import (
    AlwaysUserMuteStrategy,
    BaseUserMuteStrategy,
    FunctionCallUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start import BaseUserTurnStartStrategy, MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    default_user_turn_start_strategies,
)

from unmute_core.turn import Backchannels, InterruptionRule, word_count

Protected = Literal["greeting", "tool_calls"]
"""What the caller cannot cut off. See ``user_mute_strategies``."""
_PROTECTED: Mapping[str, type[BaseUserMuteStrategy]] = MappingProxyType(
    {
        "greeting": MuteUntilFirstBotCompleteUserMuteStrategy,
        "tool_calls": FunctionCallUserMuteStrategy,
    }
)

MIN_WORDS = 3
"""Words over the agent that cut it off. Fewer are a short answer."""


class KeepShortAnswers(BaseUserTurnStartStrategy):
    """Starts the caller's turn, but holds a short answer said over the agent.

    With the agent silent, speech or a word starts the turn at once. Over the agent,
    ``min_words`` words cut it off. Fewer stay in the aggregator, and the turn starts when
    the agent stops, without cutting anything off.

    Attributes:
        released: True while the frame that ended the agent's line starts a held turn.
            ``EndHeldTurn`` reads it.
    """

    def __init__(self, *, min_words: int = MIN_WORDS, **kwargs: Any) -> None:
        """Count words from zero, with the agent silent.

        Args:
            min_words: Words over the agent that cut it off.
            **kwargs: Pipecat's own strategy arguments.
        """
        super().__init__(**kwargs)
        self._min_words = min_words
        self._agent_speaking = False
        self._held_words = 0
        self.released = False

    async def handle_user_turn_started(self) -> None:
        """Forget the held words: they are in the turn now."""
        self._held_words = 0

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Start the turn on speech or words, unless they are a short answer over the agent.

        Args:
            frame: The frame the user aggregator got.
        """
        self.released = False
        if isinstance(frame, BotStartedSpeakingFrame):
            self._agent_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._agent_speaking = False
            if self._held_words:
                self.released = True
                await self.trigger_user_turn_started(enable_interruptions=False)
        elif isinstance(frame, VADUserStartedSpeakingFrame) and not self._agent_speaking:
            await self.trigger_user_turn_started()
            return ProcessFrameResult.STOP
        elif isinstance(frame, TranscriptionFrame | InterimTranscriptionFrame):
            return await self._words(frame)
        return ProcessFrameResult.CONTINUE

    async def _words(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame
    ) -> ProcessFrameResult:
        """Start the turn on a transcript, or hold it when it is a short answer over the agent.

        Final transcripts over one line add up, so "No wait" then "Saturday" cut it off.
        Chinese, Japanese and Thai count by ``word_count``, as they have no spaces.
        """
        words = word_count(frame.text)
        if not words:
            return ProcessFrameResult.CONTINUE
        heard = self._held_words + words
        if self._agent_speaking and heard < self._min_words:
            if isinstance(frame, TranscriptionFrame):
                self._held_words = heard
            return ProcessFrameResult.CONTINUE
        await self.trigger_user_turn_started()
        return ProcessFrameResult.STOP


class MinSpeechOverAgent(BaseUserTurnStartStrategy):
    """Starts the caller's turn over the agent only after enough speech.

    While the agent speaks, a timer starts when the caller's voice is detected. The turn
    starts, and so the agent is cut off, when the timer ends. Words the caller finishes
    before the cut are kept, so the turn holds all of them. Voice that stops first never
    cuts the agent off, and its frames never reach the strategies after this one. Its
    words are not lost: when the agent stops, they start the caller's turn without cutting
    anything off, as ``KeepShortAnswers`` does, and reach the model. A cough with no
    transcript holds nothing. A transcript that arrives after the agent stopped passes to
    Pipecat's own strategies. While the agent is silent every frame passes on, so they
    start the turn at once.

    Place it before Pipecat's default start strategies, and give it to ``EndHeldTurn``.

    Attributes:
        released: True while the frame that ended the agent's line starts a held turn.
            ``EndHeldTurn`` reads it.
    """

    def __init__(self, *, min_speech_s: float, **kwargs: Any) -> None:
        """Wait ``min_speech_s`` seconds of speech over the agent, with the agent silent.

        Args:
            min_speech_s: Seconds of speech over the agent that cut it off. The VAD's own
                ``start_secs`` counts toward it.
            **kwargs: Pipecat's own strategy arguments.
        """
        super().__init__(**kwargs)
        self._min_speech_s = min_speech_s
        self._agent_speaking = False
        self._cut = False
        self._held = False
        self._timer: asyncio.Task[None] | None = None
        self.released = False

    async def cleanup(self) -> None:
        """Stop the timer."""
        await super().cleanup()
        await self._cancel_timer()

    async def handle_user_turn_started(self) -> None:
        """Stop the timer and forget the held words: they are in the turn now."""
        self._held = False
        await self._cancel_timer()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Hold speech over the agent until it is long enough, and pass everything else on.

        Args:
            frame: The frame the user aggregator got.
        """
        self.released = False
        if isinstance(frame, BotStartedSpeakingFrame):
            self._agent_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._agent_speaking = False
            await self._cancel_timer()
            if self._held:
                self.released = True
                await self.trigger_user_turn_started(enable_interruptions=False)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._cut = False
            if self._agent_speaking:
                await self._cancel_timer()
                wait = max(0.0, self._min_speech_s - frame.start_secs)
                self._timer = self.task_manager.create_task(self._cut_after(wait), f"{self}::cut")
                return ProcessFrameResult.STOP
        elif not self._agent_speaking or self._cut:
            return ProcessFrameResult.CONTINUE
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._cancel_timer()
            return ProcessFrameResult.STOP
        elif isinstance(frame, TranscriptionFrame | InterimTranscriptionFrame):
            self._held = self._held or (
                isinstance(frame, TranscriptionFrame) and bool(frame.text.strip())
            )
            return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE

    async def _cut_after(self, seconds: float) -> None:
        """Wait, then start the turn: the caller spoke long enough over the agent."""
        await asyncio.sleep(seconds)
        self._timer = None
        self._cut = True
        await self.trigger_user_turn_started()

    async def _cancel_timer(self) -> None:
        """Stop the timer if it runs."""
        timer, self._timer = self._timer, None
        if timer is not None:
            await self.task_manager.cancel_task(timer)


class EndHeldTurn(BaseUserTurnStopStrategy):
    """Ends the turn a held answer starts, when the caller is already quiet.

    The caller finished "Yes, please." while the agent spoke, so the other stop strategies
    have already passed their moment. Without this, the turn would wait for Pipecat's
    ``user_turn_stop_timeout``.
    """

    def __init__(self, keep: KeepShortAnswers | MinSpeechOverAgent, **kwargs: Any) -> None:
        """Follow the start strategy that holds the answers.

        Args:
            keep: The start strategy that holds the answers: ``KeepShortAnswers`` or
                ``MinSpeechOverAgent``. Both say when a held turn starts, in ``released``.
            **kwargs: Pipecat's own strategy arguments.
        """
        super().__init__(**kwargs)
        self._keep = keep
        self._caller_speaking = False

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """End a turn a held answer just started, unless the caller is speaking again.

        Args:
            frame: The frame the user aggregator got.
        """
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._caller_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._caller_speaking = False
        elif (
            isinstance(frame, BotStoppedSpeakingFrame)
            and self._keep.released
            and not self._caller_speaking
        ):
            await self.trigger_user_turn_stopped()
            return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE


def interruption(
    backchannels: Backchannels,
    stop: Sequence[BaseUserTurnStopStrategy],
    *,
    min_words: int = MIN_WORDS,
    min_speech_s: float | None = None,
) -> UserTurnStrategies:
    """Return the user turn strategies for one backchannel rule.

    - ``keep``: a short answer over the agent waits for the line to end, then the agent
      answers it. A "yes" over a question is the answer to it.
    - ``drop``: Pipecat's ``MinWordsUserTurnStartStrategy``. A short answer over the agent
      is thrown away.
    - ``stop``: Pipecat's defaults. Any speech cuts the agent off. With ``min_speech_s``,
      speech over the agent must last that long first: a ``MinSpeechOverAgent`` goes
      before the defaults. Shorter speech does not cut the agent off, and its words reach
      the model once the agent stops.

    Unmute's ``minimum_words`` is ``min_words`` here (unmute@868ae23
    templates/pipecat_v1/call.py.tmpl:463-466). ``user_mute_strategies`` turns
    interruption off, or protects the greeting and tool calls.

    Args:
        backchannels: The rule.
        stop: The stop strategies, like a ``TurnAnalyzerUserTurnStopStrategy``.
        min_words: Words over the agent that cut it off, for ``keep`` and ``drop``.
        min_speech_s: Seconds of speech over the agent that cut it off, for ``stop``.
            None changes nothing.

    Raises:
        ValueError: If ``min_words`` is below 1, ``min_speech_s`` is negative, or
            ``min_speech_s`` comes with ``keep`` or ``drop``.
    """
    InterruptionRule(backchannels, min_words, min_speech_s, min_words_floor=1)
    if backchannels == "stop":
        if min_speech_s is None:
            return UserTurnStrategies(stop=list(stop))
        gate = MinSpeechOverAgent(min_speech_s=min_speech_s)
        return UserTurnStrategies(
            start=[gate, *default_user_turn_start_strategies()],
            stop=[EndHeldTurn(gate), *stop],
        )
    if backchannels == "drop":
        return UserTurnStrategies(
            start=[MinWordsUserTurnStartStrategy(min_words=min_words)], stop=list(stop)
        )
    keep = KeepShortAnswers(min_words=min_words)
    return UserTurnStrategies(start=[keep], stop=[EndHeldTurn(keep), *stop])


def user_mute_strategies(
    *, enabled: bool = True, protect: Sequence[Protected] = ()
) -> list[BaseUserMuteStrategy]:
    """Return the user mute strategies for Unmute's interruption settings.

    Give them to ``LLMUserAggregatorParams(user_mute_strategies=...)``. Ported from
    unmute@868ae23 internal/generate/pipecat_v1_build.go:1828-1898.

    Args:
        enabled: False mutes the caller while the agent speaks: no line is cut off.
        protect: What the caller cannot cut off when interruption is on: ``greeting``
            (the agent's first line) and ``tool_calls`` (a running tool).

    Raises:
        ValueError: If ``protect`` comes with ``enabled=False``, or names something else.
    """
    if unknown := [item for item in protect if item not in _PROTECTED]:
        raise ValueError(f"protect has unknown entry {unknown[0]!r}: use greeting or tool_calls")
    if not enabled:
        if protect:
            raise ValueError("protect needs interruption enabled: with it off, nothing is cut off")
        return [AlwaysUserMuteStrategy()]
    return [_PROTECTED[item]() for item in dict.fromkeys(protect)]
