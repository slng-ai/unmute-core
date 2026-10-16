"""Run a conformance scenario on the LiveKit adapter: a text-only ``AgentSession``.

No STT, TTS or room. The session's transcription output stands in for the speaker: LiveKit
sends both ``say()`` lines and model replies to it when there is no audio.
"""

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest
from livekit import rtc
from livekit.agents import NOT_GIVEN, Agent, AgentSession, RunContext, function_tool, llm
from livekit.agents.llm._realtime.openai import RealtimeModel
from livekit.agents.voice.events import CloseReason
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities, TextOutput

from tests.conformance.scenario import (
    MAIN,
    Handoff,
    Run,
    Scenario,
    agent_spec,
    handoff_tool,
    native_tool,
    new_run,
)
from tests.realtime.fake_server import FakeRealtimeServer
from unmute_core.handoff import HandoffContext
from unmute_core.livekit import (
    DevObserver,
    ModelCallLLM,
    UnmuteAgent,
    function_tools,
    max_duration,
    open_call,
    trace_session,
)


class Lines(TextOutput):
    """Collects each spoken segment as one line."""

    def __init__(self) -> None:
        super().__init__(label="conformance", next_in_chain=None)
        self.lines: list[str] = []
        self._segment = ""

    async def capture_text(self, text: str) -> None:
        """Add text to the current line."""
        self._segment += text

    def flush(self) -> None:
        """Close the current line."""
        if self._segment.strip():
            self.lines.append(self._segment.strip())
        self._segment = ""


class Silence(AudioOutput):
    """An audio sink that plays nothing, so the session has audio output, as on a call.

    A realtime model then speaks itself, and ``session.say`` needs a TTS it does not have.
    """

    def __init__(self) -> None:
        super().__init__(label="silence", capabilities=AudioOutputCapabilities(pause=False))

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Count the segment, and play nothing."""
        await super().capture_frame(frame)

    def flush(self) -> None:
        """End the segment: it has played at once."""
        super().flush()
        if self._pending_playback_count:
            self.on_playback_finished(playback_position=0.0, interrupted=False)

    def clear_buffer(self) -> None:
        """Stop the segment: it was cut off."""
        if self._pending_playback_count:
            self.on_playback_finished(playback_position=0.0, interrupted=True)


async def run_realtime(scenario: Scenario) -> dict[str, Any]:
    """Run a realtime scenario on LiveKit's own OpenAI realtime model and a fake server.

    The model's audio transcript reaches the transcription output as its spoken lines.
    """
    if not scenario.realtime:
        pytest.skip("the scenario does not run on a realtime model")
    run = new_run(scenario, realtime=True)
    async with FakeRealtimeServer(run.model) as server:
        model = RealtimeModel(
            base_url=server.http_url, api_key="test", modalities=["audio"], turn_detection=None
        )
        try:
            outcome = await run_on(scenario, run, model, settle=server.wait_idle)
        finally:
            await model.aclose()
    if server.refused:
        outcome["problems"].append(f"the server refused {server.refused} replies")
    return outcome


async def run(scenario: Scenario) -> dict[str, Any]:
    """Run every turn of the scenario and return what the agent did."""
    return await run_on(scenario, new_run(scenario))


async def run_on(
    scenario: Scenario,
    run: Run,
    realtime: RealtimeModel | None = None,
    settle: Callable[[], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Run every turn on ``UnmuteAgent``: through the scripted model call, or ``realtime``.

    ``settle`` waits after each turn until the model's server is quiet, as a caller who
    takes a moment lets the call's own work finish.
    """
    names: dict[int, str] = {}

    def build(name: str) -> UnmuteAgent:
        """Build one main agent, with a handoff tool per agent it can hand the call to."""
        part = scenario.agent(name)
        owner: list[UnmuteAgent] = []

        def handoff(tool: str) -> llm.RawFunctionTool:
            """A tool that returns the owner's ``hand_off`` to a new agent."""
            spec = part.handoff(tool)

            async def hand_off(raw_arguments: dict[str, Any], context: RunContext) -> object:
                return await transfer(spec)(owner[0])

            return function_tool(hand_off, raw_schema=handoff_tool(tool, spec.to))

        def transfer(spec: Handoff) -> Callable[[UnmuteAgent], Awaitable[object]]:
            """The owner's ``hand_off`` to a new agent, for a handoff tool or a group."""

            async def hand_off(agent: UnmuteAgent) -> object:
                return await agent.hand_off(
                    build(spec.to),
                    announce=spec.announce,
                    context=HandoffContext(**spec.context.model_dump()),
                    opens_with=spec.opens_with,
                )

            return hand_off

        def native(tool: str) -> llm.RawFunctionTool:
            """A framework tool of the main agent, answering its canned result."""

            async def call(raw_arguments: dict[str, Any], context: RunContext) -> object:
                return json.dumps(run.native(tool, scenario.native_tools, raw_arguments))

            return function_tool(call, raw_schema=native_tool(tool))

        natives = [native(tool) for tool in scenario.native_tools] if name == MAIN else []
        agent = UnmuteAgent(
            agent_spec(scenario, run, name, own_model=realtime is None),
            telemetry=None if names else run.telemetry,
            native_tools=[*(handoff(tool) for tool in part.handoffs), *natives],
            call=run.call,
            then_targets={target: transfer(spec) for target, spec in scenario.then_targets.items()},
        )
        owner.append(agent)
        names[id(agent)] = name
        return agent

    lines = Lines()
    main = build(MAIN)
    async with AgentSession(llm=realtime or NOT_GIVEN) as session:
        session.output.transcription = lines
        _record(session, run)
        closer = _watch_closing(session, names)
        if realtime is not None:
            session.output.audio = Silence()
        await session.start(main)
        await _open(session, scenario, run, lines)
        if scenario.max_duration_s is not None:
            max_duration(session, scenario.max_duration_s, call=run.call)
        await _turns(session, scenario, run, lines, settle)
        run.agent = closer[0] if closer else names.get(id(session.current_agent), "a step")
        run.ended = session._is_closing()
        if scenario.caller_hangs_up:
            await _hang_up(session)
        run.end_call(scenario.at_end)
    assert main._trace is not None and main._trace.closing is not None
    await main._trace.closing
    return run.outcome()


def _record(session: AgentSession, run: Run) -> None:
    """Send the call's dev records to the run, as a host does when the dev loop asks."""
    recorder = run.recorder("livekit")
    recorder.watch_state(run.state)
    DevObserver(recorder, controls=run.controls).attach(session)


async def _open(session: AgentSession, scenario: Scenario, run: Run, lines: Lines) -> None:
    """Open the call: pass the checks, then greet. Keep what was said before the caller."""
    run.model.script(scenario.opening)
    handle = await open_call(session, scenario.opens, run.call)
    if handle is not None:
        await handle.wait_for_playout()
    if scenario.ready_fails:
        await _closing(session)
    run.opening = list(lines.lines)
    run.model.problems += run.unused()


async def _turns(
    session: AgentSession,
    scenario: Scenario,
    run: Run,
    lines: Lines,
    settle: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Run each caller line and keep the lines said in its turn."""
    for number, turn in enumerate(scenario.turns):
        run.model.script(turn.model)
        said = len(lines.lines)
        await session.run(user_input=turn.caller)
        if settle is not None:
            await settle()
        if scenario.expect.ended and number == len(scenario.turns) - 1:
            await _closing(session)
            await _taken(run)
        run.spoken.append(lines.lines[said:])
        run.model.problems += run.unused()


async def _closing(session: AgentSession) -> None:
    """Wait until the session closes, as a failure line plays and the call hangs up."""
    async with asyncio.timeout(5):
        while not session._is_closing():
            await asyncio.sleep(0.01)


def _watch_closing(session: AgentSession, names: dict[int, str]) -> list[str]:
    """Return a list that gets the agent that held the call when the session first closes.

    A running step hands the call back to its owner while the session closes, so the agent
    is read at the request to close: it is the one that decided to end the call.
    """
    closer: list[str] = []
    close_soon = session._close_soon

    def watched(**options: Any) -> None:
        """Note the agent that holds the call, then close."""
        if not closer:
            closer.append(names.get(id(session.current_agent), "a step"))
        close_soon(**options)

    cast(Any, session)._close_soon = watched
    return closer


async def _taken(run: Run) -> None:
    """Wait until the scripted replies are taken: the call's end asks after the turn."""
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(5):
            while run.model.pending:
                await asyncio.sleep(0.01)


async def _hang_up(session: AgentSession) -> None:
    """Close the session as LiveKit does when the caller leaves the room."""
    session._close_soon(reason=CloseReason.PARTICIPANT_DISCONNECTED)
    if session._closing_task is not None:
        await session._closing_task


async def run_plain(scenario: Scenario) -> dict[str, Any]:
    """Run a scenario on a plain LiveKit ``Agent``, with the library's pieces one by one.

    ``function_tools`` gives the tools, ``ModelCallLLM`` sends the requests and
    ``trace_session`` traces the call. A scenario with tasks or handoffs needs
    ``UnmuteAgent``, so it is skipped.
    """
    if scenario.agents or scenario.agent(MAIN).tasks.tasks:
        pytest.skip("tasks and handoffs need UnmuteAgent")
    if scenario.save_values:
        pytest.skip("the values block needs UnmuteAgent")
    run = new_run(scenario)
    lines = Lines()
    async with AgentSession() as session:
        session.output.transcription = lines
        _record(session, run)
        trace = trace_session(session, run.telemetry, run.state, call=run.call)
        agent = Agent(
            instructions=scenario.instructions,
            tools=function_tools(
                run.plans[MAIN].tools,
                run.state,
                hooks=run.call,
                language=scenario.language,
                timezone=scenario.timezone,
            ),
            llm=ModelCallLLM(run.model, state=run.state, trace=trace),
        )
        await session.start(agent)
        await _open(session, scenario, run, lines)
        if scenario.max_duration_s is not None:
            max_duration(session, scenario.max_duration_s, call=run.call)
        await _turns(session, scenario, run, lines)
        run.agent = MAIN
        run.ended = session._is_closing()
        if scenario.caller_hangs_up:
            await _hang_up(session)
        run.end_call(scenario.at_end)
    assert trace.closing is not None
    await trace.closing
    return run.outcome()
