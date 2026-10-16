"""``interruption``: a short answer over the agent waits for its line, a longer one cuts it."""

from typing import Any

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)

from unmute_core.pipecat import Backchannels, interruption
from unmute_core.pipecat._interruption import MinSpeechOverAgent
from unmute_core.turn import Backchannels as TurnBackchannels

PAUSE = SleepFrame(0.1)


def said(text: str) -> list[Frame]:
    """The frames of one caller line: speech, its transcript, then quiet."""
    return [
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text, "caller", "now"),
        VADUserStoppedSpeakingFrame(),
        PAUSE,
    ]


async def play(
    frames: list[Frame],
    backchannels: Backchannels = "keep",
    stop: list[BaseUserTurnStopStrategy] | None = None,
    min_speech_s: float | None = None,
) -> tuple[Any, LLMContext]:
    """Send frames through a real user aggregator. Return the names of what came out, and the context."""
    context = LLMContext()
    stop = stop or [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.01)]
    aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            user_turn_strategies=interruption(backchannels, stop, min_speech_s=min_speech_s)
        ),
    )
    down, _ = await run_test(aggregator, frames_to_send=[*frames, SleepFrame(0.3)])
    return [type(frame).__name__ for frame in down], context


def users(context: LLMContext) -> list[str]:
    """The caller messages the context holds."""
    return [
        str(m["content"])
        for m in context.get_messages()
        if isinstance(m, dict) and m["role"] == "user"
    ]


@pytest.mark.parametrize(
    ("rule", "min_speech_s"), [("keep", None), ("stop", 1.0)], ids=["word-gate", "min-speech"]
)
async def test_a_short_answer_over_the_agent_waits_and_then_reaches_the_model(
    rule: Any, min_speech_s: float | None
) -> None:
    out, context = await play(
        [BotStartedSpeakingFrame(), PAUSE, *said("Yes, please."), BotStoppedSpeakingFrame()],
        rule,
        min_speech_s=min_speech_s,
    )

    stopped = out.index("BotStoppedSpeakingFrame")
    assert InterruptionFrame.__name__ not in out
    assert UserStartedSpeakingFrame.__name__ not in out[:stopped]
    assert LLMContextFrame.__name__ in out[stopped:]
    assert users(context) == ["Yes, please."]


@pytest.mark.parametrize("line", ["No wait, Saturday please", "ちょっと待ってください"])
async def test_a_longer_answer_over_the_agent_cuts_it_off_at_once(line: str) -> None:
    out, context = await play([BotStartedSpeakingFrame(), PAUSE, *said(line)])

    assert InterruptionFrame.__name__ in out
    assert UserStartedSpeakingFrame.__name__ in out
    assert users(context) == [line]


async def test_short_pieces_over_the_agent_add_up_to_a_cut() -> None:
    out, _ = await play([BotStartedSpeakingFrame(), PAUSE, *said("No wait"), *said("Saturday")])

    assert InterruptionFrame.__name__ in out


class NeverStops(BaseUserTurnStopStrategy):
    """Turn detection that has not decided yet: it never ends a turn."""


async def test_a_cut_is_left_to_turn_detection_when_the_caller_pauses() -> None:
    """Only a held answer ends when the agent stops. A cut waits for the caller's real end.

    The marker frame comes before the run's ``EndFrame``, which pushes what is left.
    """
    out, _ = await play(
        [
            BotStartedSpeakingFrame(),
            PAUSE,
            *said("No wait, Saturday please"),
            BotStoppedSpeakingFrame(),
            PAUSE,
            TextFrame("marker"),
        ],
        stop=[NeverStops()],
    )

    assert InterruptionFrame.__name__ in out
    assert LLMContextFrame.__name__ not in out[: out.index(TextFrame.__name__)]


async def test_a_word_with_the_agent_silent_starts_a_turn() -> None:
    out, context = await play(
        [TranscriptionFrame("Hi.", "caller", "now"), VADUserStoppedSpeakingFrame(), PAUSE]
    )

    assert UserStartedSpeakingFrame.__name__ in out
    assert users(context) == ["Hi."]


async def test_speech_with_the_agent_silent_starts_a_turn() -> None:
    out, _ = await play([VADUserStartedSpeakingFrame(), PAUSE])

    assert UserStartedSpeakingFrame.__name__ in out


async def test_the_other_rules_are_pipecats_own_strategies() -> None:
    stop = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.01)]

    dropping = interruption("drop", stop)
    stopping = interruption("stop", stop)

    assert [type(s) for s in dropping.start or []] == [MinWordsUserTurnStartStrategy]
    assert VADUserTurnStartStrategy in [type(s) for s in stopping.start or []]
    assert dropping.stop == stop


def test_backchannels_is_one_name_on_both_frameworks() -> None:
    assert Backchannels is TurnBackchannels


def test_without_min_speech_the_stop_rule_is_pipecats_default() -> None:
    strategies = interruption("stop", [NeverStops()])
    assert not any(isinstance(s, MinSpeechOverAgent) for s in strategies.start or [])


def test_min_speech_goes_before_pipecats_default_strategies() -> None:
    start = interruption("stop", [NeverStops()], min_speech_s=0.5).start or []
    assert isinstance(start[0], MinSpeechOverAgent)
    assert len(start) == 3


@pytest.mark.parametrize("rule", ["keep", "drop"])
def test_min_speech_is_refused_beside_a_word_gate(rule: Any) -> None:
    with pytest.raises(ValueError, match="stop only"):
        interruption(rule, [NeverStops()], min_speech_s=0.5)


async def test_short_speech_over_the_agent_does_not_cut_it_off() -> None:
    out, _ = await play([BotStartedSpeakingFrame(), PAUSE, *said("mm")], "stop", min_speech_s=1.0)

    assert InterruptionFrame.__name__ not in out
    assert UserStartedSpeakingFrame.__name__ not in out


async def test_long_speech_over_the_agent_starts_a_turn() -> None:
    out, context = await play(
        [
            BotStartedSpeakingFrame(),
            PAUSE,
            VADUserStartedSpeakingFrame(),
            SleepFrame(0.4),
            TranscriptionFrame("No wait", "caller", "now"),
            VADUserStoppedSpeakingFrame(),
            PAUSE,
        ],
        "stop",
        min_speech_s=0.2,
    )

    assert InterruptionFrame.__name__ in out
    assert users(context) == ["No wait"]


async def test_the_vad_start_delay_counts_toward_the_minimum() -> None:
    out, _ = await play(
        [
            BotStartedSpeakingFrame(),
            PAUSE,
            VADUserStartedSpeakingFrame(start_secs=5.0),
            SleepFrame(0.05),
            VADUserStoppedSpeakingFrame(),
            PAUSE,
        ],
        "stop",
        min_speech_s=1.0,
    )

    assert InterruptionFrame.__name__ in out


async def test_speech_while_the_agent_is_silent_passes_through() -> None:
    out, context = await play(said("Hello"), "stop", min_speech_s=5.0)

    assert UserStartedSpeakingFrame.__name__ in out
    assert users(context) == ["Hello"]


async def test_words_finished_before_the_cut_are_kept() -> None:
    out, context = await play(
        [
            BotStartedSpeakingFrame(),
            PAUSE,
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame("No wait", "caller", "now"),
            SleepFrame(0.4),
            TranscriptionFrame("Saturday", "caller", "now"),
            VADUserStoppedSpeakingFrame(),
            PAUSE,
        ],
        "stop",
        min_speech_s=0.2,
    )

    assert InterruptionFrame.__name__ in out
    assert users(context) == ["No wait Saturday"]


async def test_a_transcript_after_short_speech_stops_is_kept_for_the_model() -> None:
    out, context = await play(
        [
            BotStartedSpeakingFrame(),
            PAUSE,
            VADUserStartedSpeakingFrame(),
            TranscriptionFrame("mm", "caller", "now"),
            VADUserStoppedSpeakingFrame(),
            PAUSE,
            TranscriptionFrame("hm", "caller", "now"),
            PAUSE,
            BotStoppedSpeakingFrame(),
        ],
        "stop",
        min_speech_s=1.0,
    )

    assert InterruptionFrame.__name__ not in out
    assert LLMContextFrame.__name__ in out[out.index("BotStoppedSpeakingFrame") :]
    assert users(context) == ["mm hm"]


async def test_a_cough_with_no_transcript_holds_nothing() -> None:
    out, context = await play(
        [
            BotStartedSpeakingFrame(),
            PAUSE,
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            PAUSE,
            BotStoppedSpeakingFrame(),
            PAUSE,
        ],
        "stop",
        min_speech_s=1.0,
    )

    assert UserStartedSpeakingFrame.__name__ not in out
    assert LLMContextFrame.__name__ not in out
    assert users(context) == []


async def test_a_transcript_after_the_agent_stopped_goes_to_the_default_strategies() -> None:
    out, context = await play(
        [
            BotStartedSpeakingFrame(),
            PAUSE,
            VADUserStartedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            PAUSE,
            TranscriptionFrame("Saturday", "caller", "now"),
            PAUSE,
        ],
        "stop",
        min_speech_s=1.0,
    )

    assert UserStartedSpeakingFrame.__name__ in out
    assert users(context) == ["Saturday"]
