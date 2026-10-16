"""Run a conformance scenario on the Pipecat adapter: a real pipeline with no transport.

A recorder after the LLM keeps each model response and each ``TTSSpeakFrame`` as one
spoken line. Pipecat's own TTS service follows it, with silence for a voice, so a line
joins the context the way it does on a call. An output after it reports the bot starting
and stopping to speak, as a transport does. A turn is over when the model has answered
every scripted request and a marker frame, queued after them, has come out of the TTS.

The caller types each line, as RTVI sends typed text. With ``voice=True`` the caller speaks
it instead: VAD frames and a transcript go through the user aggregator, which decides the
turn as on a call.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner

from tests.conformance.scenario import (
    MAIN,
    Handoff,
    Run,
    Scenario,
    ScriptedModel,
    agent_spec,
    handoff_tool,
    native_tool,
    new_run,
)
from tests.realtime.fake_server import FakeRealtimeServer
from unmute_core.handoff import SUMMARY_SITE, HandoffContext
from unmute_core.pipecat import (
    DevObserver,
    ModelCallLLMService,
    TelemetryObserver,
    UnmuteFlows,
    max_duration,
    open_call,
)
from unmute_core.pipecat._tools import function_schema


@dataclass
class Marker(DataFrame):
    """A frame that marks the end of a turn. A data frame keeps its place in the queue.

    It goes down the pipeline, up again, and down a second time, marked ``back``. Then
    what each processor pushed on the way, up or down, has been handled. One example is
    the commit of a line code said, which the LLM service sends once the bot stops.
    """

    turn: int = 0
    back: bool = False


class TurnStart(FrameProcessor):
    """Sends a marker that came up the pipeline back down. It is first in the pipeline."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Turn a marker going up around, and pass every other frame on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, Marker) and direction == FrameDirection.UPSTREAM:
            await self.push_frame(Marker(turn=frame.turn, back=True))
            return
        await self.push_frame(frame, direction)


class TurnEnd(FrameProcessor):
    """Sends a new marker back up, and notes one on its way back. It is last in the pipeline."""

    def __init__(self, speaker: "Speaker") -> None:
        super().__init__()
        self.speaker = speaker

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Turn a new marker around, note a marker that is back, and pass every other frame."""
        await super().process_frame(frame, direction)
        if not isinstance(frame, Marker):
            await self.push_frame(frame, direction)
        elif frame.back:
            self.speaker.marker = frame.turn
        else:
            await self.push_frame(Marker(turn=frame.turn), FrameDirection.UPSTREAM)


class Speaker(FrameProcessor):
    """Records what the agent says, in order, and passes every frame on.

    With a realtime model there is no TTS, so it reports a spoken reply upstream itself, as
    the bot starting and then stopping to speak.
    """

    def __init__(self, *, realtime: bool = False) -> None:
        super().__init__()
        self.realtime = realtime
        self.lines: list[str] = []
        self.responses = 0
        self.marker = -1
        self._reply: str | None = None

    @property
    def replying(self) -> bool:
        """Whether a response has started and not ended yet."""
        return self._reply is not None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Keep spoken text, count finished responses, and note the latest marker."""
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._reply = ""
        elif isinstance(frame, LLMTextFrame) and self._reply is not None:
            self._reply += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._reply and self._reply.strip():
                self.lines.append(self._reply.strip())
                if self.realtime:
                    await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
                    await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._reply = None
            self.responses += 1
        elif isinstance(frame, TTSSpeakFrame):
            self.lines.append(frame.text.strip())
            if self.realtime:
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
                await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        await self.push_frame(frame, direction)


class SilentTTS(TTSService):
    """Pipecat's TTS service with silence for a voice: each sentence is 20 ms of it."""

    def __init__(self) -> None:
        super().__init__(
            sample_rate=16000,
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=0.05,
            settings=TTSSettings(model=None, voice=None, language=None),
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None]:
        yield TTSAudioRawFrame(b"\x00" * 640, 16000, 1, context_id=context_id)


class Output(FrameProcessor):
    """Plays the TTS's speech as an output transport does.

    The bot starts to speak at a line's first audio and stops once the TTS stops.
    """

    def __init__(self) -> None:
        super().__init__()
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame on but the audio."""
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            if not self._speaking:
                self._speaking = True
                await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            return
        await self.push_frame(frame, direction)
        if isinstance(frame, TTSStoppedFrame) and self._speaking:
            self._speaking = False
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)


def _replies(model: ScriptedModel) -> int:
    """Return the requests that end in a reply. A finish check or a summary has none."""
    return sum(
        1
        for request in model.requests
        if request.tool_choice != "required" and request.site != SUMMARY_SITE
    )


async def _until(check: Callable[[], bool]) -> None:
    """Wait until ``check`` holds, for at most five seconds.

    It checks on every step of the event loop. A wait that can pass too early then fails
    on every run, not once in a hundred.
    """
    async with asyncio.timeout(5):
        while not check():
            await asyncio.sleep(0)


async def _tools_done(llm: LLMService[Any]) -> None:
    """Wait until every tool call the model made has returned.

    A handler queues what it says before it returns, like the line a task step opens with
    after the pipeline flush. A turn ends only then, so its marker queues behind those lines.
    Pipecat holds the running calls in ``_function_call_tasks`` from before the reply's end
    frame until each handler returns.
    """
    await _until(lambda: not llm._function_call_tasks)


async def _mark(worker: PipelineWorker, speaker: Speaker, turn: int) -> None:
    """End a turn: queue its marker and wait until it has passed and no reply is open.

    A realtime service starts a reply in the pipeline's order, but its text and its end come
    on the service's own task. They can reach the speaker after the marker.
    """
    await worker.queue_frame(Marker(turn=turn))
    await _until(lambda: speaker.marker == turn and not speaker.replying)


def _caller(text: str, *, voice: bool) -> list[Frame]:
    """Return the frames of one caller line: typed, or spoken and transcribed."""
    if not voice:
        return [LLMMessagesAppendFrame([{"role": "user", "content": text}], run_llm=True)]
    return [
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text, "caller", time_now_iso8601(), finalized=True),
        VADUserStoppedSpeakingFrame(),
    ]


async def run(scenario: Scenario, *, voice: bool = False) -> dict[str, Any]:
    """Run every turn of the scenario and return what the agent did.

    Args:
        scenario: The scenario.
        voice: Speak each caller line instead of typing it.
    """
    run = new_run(scenario)
    context = LLMContext()
    stop = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.01)
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=UserTurnStrategies(stop=[stop])),
    )
    llm = ModelCallLLMService(
        run.model,
        state=run.state,
        settings=OpenAILLMService.Settings(
            model="scripted", system_instruction=scenario.instructions
        ),
    )

    async def caller(text: str) -> None:
        """Type or speak one caller line."""
        for frame in _caller(text, voice=voice):
            await worker.queue_frame(frame)

    async def answered() -> None:
        """Wait until the model has answered every scripted request of the turn."""
        await _until(lambda: not run.model.pending and speaker.responses == _replies(run.model))

    speaker = Speaker()
    worker, flows = _run_on(scenario, run, llm, context, aggregators, speaker)
    return await _turns(scenario, run, worker, flows, speaker, context, caller, answered)


async def run_realtime(scenario: Scenario) -> dict[str, Any]:
    """Run a realtime scenario on Pipecat's OpenAI realtime service and a fake server.

    The server plays the caller's speech as server VAD does. The model's audio transcript
    is its spoken text.
    """
    if not scenario.realtime:
        pytest.skip("the scenario does not run on a realtime model")
    run = new_run(scenario, realtime=True)
    async with FakeRealtimeServer(run.model) as server:
        context = LLMContext()
        aggregators = LLMContextAggregatorPair(context)
        llm = OpenAIRealtimeLLMService(
            api_key="test",
            base_url=server.ws_url,
            settings=OpenAIRealtimeLLMService.Settings(
                session_properties=events.SessionProperties(output_modalities=["audio"]),
                system_instruction=scenario.instructions,
            ),
        )

        async def answered() -> None:
            """Wait until the server has answered, the client has gone quiet and read it all.

            The client reads each ``response.done`` on its own task, so a busy machine can
            still be behind the server after the quiet time.
            """
            await server.wait_idle(quiet=0.3)
            await _until(lambda: speaker.responses >= server.done)

        speaker = Speaker(realtime=True)
        worker, flows = _run_on(scenario, run, llm, context, aggregators, speaker)
        outcome = await _turns(
            scenario, run, worker, flows, speaker, context, server.caller_says, answered
        )
    if server.refused:
        outcome["problems"].append(f"the server refused {server.refused} replies")
    return outcome


def _run_on(
    scenario: Scenario,
    run: Run,
    llm: LLMService[Any],
    context: LLMContext,
    aggregators: LLMContextAggregatorPair,
    speaker: Speaker,
) -> tuple[PipelineWorker, UnmuteFlows]:
    """Build the call's ``UnmuteFlows`` and pipeline on ``llm``, ready to run."""
    own_model = isinstance(llm, ModelCallLLMService)

    def native(name: str) -> list[FunctionSchema]:
        """The handoff tools of one main agent, each handing the call on through ``flows``."""

        def handoff(tool: str) -> FunctionSchema:
            """A tool whose handler gives the call to another agent."""
            spec = scenario.agent(name).handoff(tool)
            return function_schema(handoff_tool(tool, spec.to), transfer(spec))

        def framework(tool: str) -> FunctionSchema:
            """A framework tool of the main agent, answering its canned result."""

            async def answer(params: FunctionCallParams) -> None:
                result = run.native(tool, scenario.native_tools, dict(params.arguments))
                await params.result_callback(result)

            return function_schema(native_tool(tool), answer)

        natives = [framework(tool) for tool in scenario.native_tools] if name == MAIN else []
        return [*(handoff(tool) for tool in scenario.agent(name).handoffs), *natives]

    def transfer(spec: Handoff) -> Callable[[FunctionCallParams], Awaitable[None]]:
        """The handler of a handoff to another agent, for a handoff tool or a group."""

        async def hand_off(params: FunctionCallParams) -> None:
            await flows.hand_off(
                params,
                agent_spec(scenario, run, spec.to, own_model=own_model),
                native_tools=native(spec.to),
                announce=spec.announce,
                context=HandoffContext(**spec.context.model_dump()),
                opens_with=spec.opens_with,
            )

        return hand_off

    flows = UnmuteFlows(
        agent_spec(scenario, run, MAIN, own_model=own_model),
        llm=llm,
        context=context,
        aggregators=aggregators,
        call=run.call,
        telemetry=run.telemetry,
        native_tools=native(MAIN),
        then_targets={target: transfer(spec) for target, spec in scenario.then_targets.items()},
    )
    flows.install()
    run.telemetry.start_call()
    observer = TelemetryObserver(run.telemetry, model="scripted", provider="scripted", flows=flows)
    recorder = run.recorder("pipecat")
    recorder.watch_state(run.state)
    flows.listen(recorder.task_event)
    delegates = {delegate.name for plan in run.plans.values() for delegate in plan.delegates()}
    voice = [] if speaker.realtime else [SilentTTS(), Output()]
    worker = PipelineWorker(
        Pipeline(
            [
                TurnStart(),
                aggregators.user(),
                llm,
                speaker,
                *voice,
                aggregators.assistant(),
                TurnEnd(speaker),
            ]
        ),
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[observer, DevObserver(recorder, handoffs=delegates, controls=scenario.controls)],
    )
    return worker, flows


async def _turns(
    scenario: Scenario,
    run: Run,
    worker: PipelineWorker,
    flows: UnmuteFlows,
    speaker: Speaker,
    context: LLMContext,
    caller: Callable[[str], Awaitable[None]],
    answered: Callable[[], Awaitable[None]],
) -> dict[str, Any]:
    """Open the call, run each caller line, and return what the agent did.

    On OpenAI's realtime service the call opens once the session is set up. A reply asked
    for before then waits for the setup, and so starts after the opening's marker.
    """
    model = run.model
    llm = flows.llm
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        model.script(scenario.opening)
        realtime = llm if run.realtime else None
        if isinstance(llm, OpenAIRealtimeLLMService):
            await _until(lambda: llm._api_session_ready)
        await open_call(worker, scenario.opens, run.call, context=context, llm=realtime)
        if scenario.max_duration_s is not None:
            max_duration(worker, scenario.max_duration_s, call=run.call)
        if scenario.ready_fails:
            await _until(running.done)
        else:
            await answered()
            await _tools_done(llm)
            await _mark(worker, speaker, -2)
        run.opening = list(speaker.lines)
        model.problems += run.unused()
        for number, turn in enumerate(scenario.turns):
            model.script(turn.model)
            said = len(speaker.lines)
            await caller(turn.caller)
            await answered()
            if scenario.expect.ended and number == len(scenario.turns) - 1:
                await _until(running.done)
            else:
                await _tools_done(llm)
                await _mark(worker, speaker, number)
            run.spoken.append(speaker.lines[said:])
        if flows._run is None:
            run.agent = flows._name
        else:
            run.agent = "a step"
        run.ended = running.done()
        if scenario.caller_hangs_up:
            run.call.caller_left()
        run.end_call(scenario.at_end)
        await flows.settle()
    finally:
        if not running.done():
            await worker.queue_frame(EndFrame())
        await asyncio.wait_for(running, 5)
    await run.call.end("ended")
    await run.telemetry.close(state=run.state)
    return run.outcome()
