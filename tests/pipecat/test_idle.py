"""``IdleNudgeObserver`` reads the agent's state from frames and speaks the nudges."""

import asyncio
from typing import Any, cast

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSSpeakFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from unmute_core.call import IDLE, CallHooks
from unmute_core.idle import IdleNudges
from unmute_core.pipecat import IdleNudgeObserver

CONFIG = IdleNudges(
    first_nudge_text="Are you still there?",
    second_nudge_text="Still here when you are.",
    final_hangup_text="Goodbye.",
)
OTHER = FrameProcessor()


class Worker:
    """Plays each queued line at once, as start and stop frames, and records the end."""

    def __init__(self) -> None:
        self.observer: IdleNudgeObserver | None = None
        self.said: list[str] = []
        self.ended = False

    async def queue_frame(self, frame: Frame) -> None:
        assert isinstance(frame, TTSSpeakFrame) and self.observer is not None
        self.said.append(frame.text)
        await push(self.observer, BotStartedSpeakingFrame())
        await push(self.observer, BotStoppedSpeakingFrame())

    async def end(self) -> None:
        self.ended = True


async def push(observer: IdleNudgeObserver, frame: Frame) -> None:
    await observer.on_push_frame(
        FramePushed(
            source=OTHER,
            destination=OTHER,
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )
    )


async def _no_wait(_seconds: float) -> None:
    await asyncio.sleep(0)


async def _settle() -> None:
    for _ in range(30):
        await asyncio.sleep(0)


def _observer() -> tuple[IdleNudgeObserver, Worker]:
    worker = Worker()
    observer = IdleNudgeObserver(CONFIG, cast(PipelineWorker, cast(Any, worker)))
    observer.controller._sleep = _no_wait
    worker.observer = observer
    return observer, worker


async def test_a_quiet_caller_hears_the_three_lines_then_the_call_ends() -> None:
    observer, worker = _observer()

    await push(observer, BotStartedSpeakingFrame())
    await push(observer, BotStoppedSpeakingFrame())
    await _settle()

    assert worker.said == ["Are you still there?", "Still here when you are.", "Goodbye."]
    assert worker.ended


async def test_the_idle_hang_up_is_noted_on_the_call_before_it_ends() -> None:
    worker = Worker()
    call = CallHooks()
    observer = IdleNudgeObserver(CONFIG, cast(PipelineWorker, cast(Any, worker)), call=call)
    observer.controller._sleep = _no_wait
    worker.observer = observer

    await push(observer, BotStartedSpeakingFrame())
    await push(observer, BotStoppedSpeakingFrame())
    await _settle()

    assert worker.ended
    assert (await call.end("ended")).reason == IDLE


async def test_a_running_tool_holds_the_wait_until_its_result() -> None:
    observer, worker = _observer()
    gate = asyncio.Event()

    async def held(_seconds: float) -> None:
        await gate.wait()

    observer.controller._sleep = held
    await push(observer, LLMFullResponseStartFrame())
    await push(observer, FunctionCallInProgressFrame("lookup", "c1", {}))
    await push(observer, LLMFullResponseEndFrame())
    await push(observer, BotStartedSpeakingFrame())
    await push(observer, BotStoppedSpeakingFrame())
    gate.set()
    await _settle()
    assert worker.said == []

    await push(observer, FunctionCallResultFrame("lookup", "c1", {}, {"found": True}))
    await push(observer, LLMFullResponseStartFrame())
    await push(observer, BotStartedSpeakingFrame())
    await push(observer, LLMFullResponseEndFrame())
    await push(observer, BotStoppedSpeakingFrame())
    await _settle()
    assert worker.said[:1] == ["Are you still there?"]


async def test_the_callers_words_start_the_wait_over() -> None:
    observer, worker = _observer()
    gate = asyncio.Event()

    async def held(_seconds: float) -> None:
        await gate.wait()

    observer.controller._sleep = held
    await push(observer, BotStartedSpeakingFrame())
    await push(observer, BotStoppedSpeakingFrame())
    await _settle()
    await push(observer, InterimTranscriptionFrame("hang on", "caller", "now"))
    gate.set()
    await _settle()

    assert worker.said == []
    observer.controller.close()
