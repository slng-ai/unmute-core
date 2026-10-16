"""Opening a Pipecat call: the phone call's facts, and the greeting's frames."""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.turns.user_mute import (
    AlwaysUserMuteStrategy,
    FunctionCallUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pydantic import BaseModel

from tests.hooks.helpers import checks, on_end
from tests.outputs import AnyResult
from tests.pipecat.harness import until
from unmute_core.agent import AgentSpec
from unmute_core.call import (
    AGENT_ENDED,
    CALLER_HUNG_UP,
    ENDED,
    FAILURE_LINE,
    MAX_DURATION,
    NOTES_HEADER,
    CallEnd,
    CallHooks,
    EventTool,
)
from unmute_core.greeting import GREETING_HOLD_SECONDS, GREETING_INSTRUCTION, Greeting
from unmute_core.hooks import Hooks, PromptSuffix
from unmute_core.pipecat import (
    GreetingHold,
    UnmuteFlows,
    attach_call,
    fill_prompt,
    greeting_frames,
    interruption,
    max_duration,
    open_call,
    telephony_call_info,
    user_mute_strategies,
)
from unmute_core.state import CallState
from unmute_core.tools import local_tool

CALLER = "+34600111222"
TRUNK = "+34910000000"


def test_an_inbound_call_reads_its_ids_from_the_handshake_and_numbers_from_the_body():
    call_data = {
        "call_id": "CA1",
        "stream_id": "MZ1",
        "body": {"from_number": CALLER, "to_number": TRUNK},
    }
    call = telephony_call_info(call_data, agent_id="salon", session_id="s-1")
    assert call.model_dump(exclude_none=True) == {
        "agent_id": "salon",
        "session_id": "s-1",
        "channel": "phone",
        "direction": "inbound",
        "calling_number": CALLER,
        "called_number": TRUNK,
        "remote_number": CALLER,
        "carrier_call_id": "CA1",
    }


def test_an_outbound_call_keeps_the_trunk_as_its_own_number():
    call_data = {"call_id": "CA2", "body": {"direction": "outbound", "to_number": CALLER}}
    call = telephony_call_info(call_data, trunk_number=TRUNK)
    assert (call.calling_number, call.called_number) == (TRUNK, CALLER)


def test_telnyx_and_exotel_numbers_come_from_the_handshake_itself():
    from pipecat.runner.types import CallData, ExotelCallData, TelnyxCallData

    telnyx = TelnyxCallData.model_validate(
        {"call_id": "v3:abc", "stream_id": "s1", "from": CALLER, "to": TRUNK}
    )
    call = telephony_call_info(telnyx)
    assert (call.calling_number, call.called_number, call.carrier_call_id) == (
        CALLER,
        TRUNK,
        "v3:abc",
    )
    exotel = ExotelCallData.model_validate({"call_id": "e1", "from": CALLER, "to": TRUNK})
    assert telephony_call_info(exotel).remote_number == CALLER
    twilio = CallData.model_validate(
        {"call_id": "CA1", "body": {"from_number": CALLER}, "from": CALLER, "to": None}
    )
    assert telephony_call_info(twilio, trunk_number=TRUNK).called_number == TRUNK


def test_the_greeting_frames_say_the_line_ask_for_one_or_wait():
    [line] = greeting_frames(Greeting(text="Hello!"), None)
    assert isinstance(line, TTSSpeakFrame) and line.text == "Hello!"
    [ask] = greeting_frames(Greeting(), None)
    assert isinstance(ask, LLMMessagesAppendFrame) and ask.run_llm
    assert ask.messages == [{"role": "developer", "content": GREETING_INSTRUCTION}]
    assert greeting_frames(Greeting(speaks_first="user"), None) == []


class Worker:
    """A pipeline worker that queues frames and reports its end."""

    def __init__(self) -> None:
        self.queued: list[Any] = []
        self.handlers: dict[str, Any] = {}

    async def queue_frame(self, frame: Any) -> None:
        self.queued.append(frame)

    def add_event_handler(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler


async def test_max_duration_ends_the_pipeline_unless_it_finished_first():
    worker = Worker()
    max_duration(cast(Any, worker), 0.01)
    await until(lambda: worker.queued)
    assert [type(frame) for frame in worker.queued] == [EndFrame]
    finished = Worker()
    max_duration(cast(Any, finished), 0.01)
    await finished.handlers["on_pipeline_finished"](finished, None)
    await asyncio.sleep(0.05)  # past the 0.01 s limit, which would end the pipeline
    assert finished.queued == []


async def test_max_duration_notes_the_end_so_the_call_ends_with_that_reason():
    worker = Pipeline()
    ended: list[CallEnd] = []
    call = CallHooks(hooks=on_end(ended.append))
    attach_call(cast(Any, worker), call, LLMContext())
    max_duration(cast(Any, worker), 0.01, call=call)
    await until(lambda: worker.queued)
    assert [type(frame) for frame in worker.queued] == [EndFrame]
    await worker.emit("on_pipeline_finished", worker, EndFrame())
    assert ended[0].reason == MAX_DURATION


async def test_a_failure_after_the_end_was_asked_keeps_the_asked_reason():
    worker = Pipeline()
    ended: list[CallEnd] = []
    call = CallHooks(hooks=on_end(ended.append))
    attach_call(cast(Any, worker), call, LLMContext())
    call.end_requested(AGENT_ENDED)
    llm = OpenAILLMService(api_key="k", settings=OpenAILLMService.Settings(model="m"))
    await llm.set_usable(False)
    await worker.emit("on_pipeline_error", worker, ErrorFrame("down", processor=llm))
    await asyncio.sleep(0.01)  # time for a wrong failure line to start
    assert worker.queued == []
    await worker.emit("on_pipeline_finished", worker, EndFrame())
    assert ended[0].reason == AGENT_ENDED


def test_unmutes_interruption_settings_map_onto_mute_strategies():
    [always] = user_mute_strategies(enabled=False)
    assert isinstance(always, AlwaysUserMuteStrategy)
    greeting, tools = user_mute_strategies(protect=["greeting", "tool_calls"])
    assert isinstance(greeting, MuteUntilFirstBotCompleteUserMuteStrategy)
    assert isinstance(tools, FunctionCallUserMuteStrategy)
    assert user_mute_strategies() == []
    with pytest.raises(ValueError, match="protect needs interruption enabled"):
        user_mute_strategies(enabled=False, protect=["greeting"])
    with pytest.raises(ValueError, match="unknown entry 'music'"):
        user_mute_strategies(protect=cast(Any, ["music"]))


def test_minimum_words_sets_how_many_words_cut_the_agent_off():
    [start] = interruption("drop", [], min_words=5).start or []
    assert isinstance(start, MinWordsUserTurnStartStrategy) and start._min_words == 5
    with pytest.raises(ValueError, match="1 or more"):
        interruption("keep", [], min_words=0)


async def test_the_greeting_hold_lets_go_as_the_greeting_starts_or_ends():
    hold = GreetingHold()
    assert await hold.process_frame(UserStartedSpeakingFrame())
    assert not await hold.process_frame(BotStartedSpeakingFrame())
    protected = GreetingHold(interruptible=False)
    assert await protected.process_frame(BotStartedSpeakingFrame())
    assert not await protected.process_frame(BotStoppedSpeakingFrame())
    failed = GreetingHold()
    assert not await failed.process_frame(ErrorFrame("tts down"))
    released = GreetingHold()
    released.release()
    assert not await released.process_frame(UserStartedSpeakingFrame())


async def test_the_greeting_hold_lets_go_after_its_longest_wait_once_armed():
    hold = GreetingHold(max_hold_s=0.05)
    await asyncio.sleep(0.1)
    assert await hold.process_frame(UserStartedSpeakingFrame())
    hold.arm()
    assert await hold.process_frame(UserStartedSpeakingFrame())
    await asyncio.sleep(0.1)
    assert not await hold.process_frame(UserStartedSpeakingFrame())
    assert not await hold.process_frame(UserStartedSpeakingFrame())


class Pipeline(Worker):
    """A worker that queues several frames and runs several handlers per event."""

    def __init__(self) -> None:
        super().__init__()
        self.handlers: dict[str, list[Any]] = {}
        self.composed = 0

    def _compose_system_instruction(self) -> None:
        self.composed += 1

    def add_event_handler(self, name: str, handler: Any) -> None:
        self.handlers.setdefault(name, []).append(handler)

    async def queue_frames(self, frames: list[Any]) -> None:
        self.queued.extend(frames)

    async def emit(self, name: str, *args: Any) -> None:
        for handler in self.handlers.get(name, []):
            await handler(*args)


async def test_a_live_model_waiting_for_the_caller_lets_the_caller_speak():
    from tests.pipecat.test_realtime import live

    worker = Pipeline()
    hold = GreetingHold()
    llm = live()
    assert await open_call(
        cast(Any, worker),
        Greeting(speaks_first="user"),
        CallHooks(),
        context=LLMContext(),
        llm=llm,
        hold=hold,
    )
    assert [type(frame).__name__ for frame in worker.queued] == ["LLMRunFrame"]
    assert not await hold.process_frame(UserStartedSpeakingFrame())


async def test_open_call_queues_the_greeting_once_the_checks_pass():
    worker = Pipeline()
    hold = GreetingHold()
    call = CallHooks()
    assert await open_call(
        cast(Any, worker), Greeting(speaks_first="user"), call, context=LLMContext(), hold=hold
    )
    assert worker.queued == []
    assert not await hold.process_frame(UserStartedSpeakingFrame())

    worker = Pipeline()
    late = GreetingHold(max_hold_s=0.05)
    await open_call(
        cast(Any, worker), Greeting(text="Hi."), CallHooks(), context=LLMContext(), hold=late
    )
    assert await late.process_frame(UserStartedSpeakingFrame())
    await asyncio.sleep(0.1)
    assert not await late.process_frame(UserStartedSpeakingFrame())

    worker = Pipeline()
    assert await open_call(
        cast(Any, worker), Greeting(text="Hi."), CallHooks(), context=LLMContext()
    )
    assert [frame.text for frame in worker.queued] == ["Hi."]


async def test_a_failed_check_says_the_line_then_ends_the_pipeline():
    worker = Pipeline()

    async def broken() -> None:
        raise RuntimeError("down")

    call = CallHooks(hooks=checks(broken))
    assert not await open_call(cast(Any, worker), Greeting(text="Hi."), call, context=LLMContext())
    assert isinstance(worker.queued[0], TTSSpeakFrame)
    assert worker.queued[0].text == FAILURE_LINE
    assert isinstance(worker.queued[1], EndFrame)


async def test_an_unusable_service_fails_the_call_and_a_usable_one_does_not():
    worker = Pipeline()
    attach_call(cast(Any, worker), CallHooks(), LLMContext())
    tts = SimpleNamespace(is_usable=True)
    await worker.emit("on_pipeline_error", worker, ErrorFrame("blip", processor=cast(Any, tts)))
    await asyncio.sleep(0)
    assert worker.queued == []
    llm = OpenAILLMService(api_key="k", settings=OpenAILLMService.Settings(model="m"))
    await llm.set_usable(False)
    await worker.emit("on_pipeline_error", worker, ErrorFrame("down", processor=llm))
    await until(lambda: len(worker.queued) == 2)
    assert [type(frame) for frame in worker.queued] == [TTSSpeakFrame, EndFrame]


async def test_the_pipelines_end_ends_the_call_with_the_transcript():
    worker = Pipeline()
    ended: list[CallEnd] = []
    context = LLMContext(
        cast(
            Any,
            [
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
    )
    attach_call(cast(Any, worker), CallHooks(hooks=on_end(ended.append)), context)
    await worker.emit("on_pipeline_finished", worker, EndFrame(reason="runner exiting"))
    assert ended[0].reason == ENDED
    assert ended[0].transcript == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


async def test_a_plain_llm_waits_for_the_tools_on_events_and_reads_their_notes():
    async def lookup() -> dict:
        """Look the caller up."""
        return {"name": "Ana"}

    call = CallHooks(
        hooks=Hooks(
            EventTool(local_tool(lookup, output=AnyResult), ["first_user_message"], note=True)
        )
    )
    context = LLMContext(cast(Any, [{"role": "user", "content": "hi"}]))
    worker = Pipeline()
    llm = Pipeline()
    attach_call(cast(Any, worker), call, context, llm=cast(Any, llm))
    await llm.emit("on_before_process_frame", llm, LLMContextFrame(context))
    await llm.emit("on_before_process_frame", llm, LLMContextFrame(context))
    messages = cast(list[dict[str, Any]], context.get_messages())
    assert [m["role"] for m in messages] == ["user", "system"]
    assert messages[-1]["content"].startswith(NOTES_HEADER)


async def test_open_call_sets_the_hold_from_the_greeting_and_attaches_once():
    worker = Pipeline()
    hold = GreetingHold()
    call = CallHooks()
    await open_call(
        cast(Any, worker),
        Greeting(text="Hi.", interruptible=False),
        call,
        context=LLMContext(),
        hold=hold,
    )
    assert not hold.interruptible
    assert await hold.process_frame(BotStartedSpeakingFrame())
    attach_call(cast(Any, Pipeline()), call, LLMContext())
    assert len(worker.handlers["on_pipeline_error"]) == 1


async def test_flows_give_a_native_llm_the_main_prompt_with_a_value_saved_since():
    class Shop(BaseModel):
        closes_at: str | None = None

    state = CallState(Shop)
    context = LLMContext()
    llm = OpenAILLMService(api_key="k", settings=OpenAILLMService.Settings(model="m"))
    flows = UnmuteFlows(
        AgentSpec("We close at {{closes_at}}."),
        llm=llm,
        context=context,
        aggregators=LLMContextAggregatorPair(context),
        state=state,
    )
    state.save({"closes_at": "18:00"})
    await flows._on_process(llm, LLMContextFrame(context))
    assert llm._settings.system_instruction == "We close at 18:00."


async def test_a_main_prompt_that_cannot_be_filled_leaves_the_request_going(caplog):
    class Shop(BaseModel):
        closes_at: str | None = None

    context = LLMContext(cast(Any, [{"role": "user", "content": "hi"}]))
    llm = OpenAILLMService(api_key="k", settings=OpenAILLMService.Settings(model="m"))
    flows = UnmuteFlows(
        AgentSpec("We close at {{nowhere}}."),
        llm=llm,
        context=context,
        aggregators=LLMContextAggregatorPair(context),
        state=CallState(Shop),
    )
    with caplog.at_level(logging.ERROR):
        await flows._on_process(llm, LLMContextFrame(context))
    assert "could not be refreshed" in caplog.text
    assert flows._replies.request == context.get_messages()


def test_the_greeting_hold_lets_go_after_the_shared_hold_time_by_default():
    assert GreetingHold().max_hold_s == GREETING_HOLD_SECONDS


class Transport:
    """A transport that keeps its event handlers."""

    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}

    def add_event_handler(self, name: str, handler: Any) -> None:
        self.handlers.setdefault(name, []).append(handler)

    async def disconnect(self) -> None:
        for handler in self.handlers.get("on_client_disconnected", []):
            await handler(self, object())


async def test_a_transport_that_loses_its_client_notes_the_caller_hung_up():
    call, transport = CallHooks(), Transport()
    attach_call(cast(Any, Pipeline()), call, LLMContext(), transport=cast(Any, transport))
    await transport.disconnect()
    assert (await call.end("ended")).reason == CALLER_HUNG_UP


async def test_a_call_that_ended_for_another_reason_keeps_it_when_the_client_leaves():
    call, transport = CallHooks(), Transport()
    attach_call(cast(Any, Pipeline()), call, LLMContext(), transport=cast(Any, transport))
    call.end_requested(AGENT_ENDED)
    await transport.disconnect()
    assert (await call.end("ended")).reason == AGENT_ENDED


async def test_model_hooks_with_no_llm_to_run_on_warn_once_the_pipeline_starts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = Pipeline()
    attach_call(cast(Any, worker), CallHooks(hooks=Hooks(PromptSuffix("Be brief."))), LLMContext())
    with caplog.at_level(logging.WARNING):
        await worker.emit("on_pipeline_started", worker, None)
    assert "prompt hooks" in caplog.text
    assert "pass llm= to attach_call" in caplog.text


async def test_fill_prompt_gives_a_plain_llm_the_prompt_with_the_values_saved_since():
    class Shop(BaseModel):
        closes_at: str | None = None

    state = CallState(Shop)
    llm, worker = Pipeline(), Pipeline()
    await fill_prompt(cast(Any, llm), cast(Any, worker), "We close at {{closes_at}}.", state)
    state.save({"closes_at": "18:00"})
    await asyncio.sleep(0)
    sent = [frame.delta.system_instruction for frame in llm.queued]
    assert sent == ["We close at none recorded yet..", "We close at 18:00."]
    await worker.emit("on_pipeline_finished", worker, EndFrame())
    state.save({"closes_at": "19:00"})
    await asyncio.sleep(0)
    assert len(llm.queued) == 2


@pytest.mark.parametrize("flows", [False, True])
async def test_open_call_warns_about_model_hooks_only_when_nothing_runs_them(
    flows: bool, caplog: pytest.LogCaptureFixture
) -> None:
    class Shop(BaseModel):
        closes_at: str | None = None

    call = CallHooks(CallState(Shop), hooks=Hooks(PromptSuffix("Be brief.")))
    context = LLMContext()
    if flows:
        UnmuteFlows(
            AgentSpec("Orders."),
            llm=OpenAILLMService(api_key="k", settings=OpenAILLMService.Settings(model="m")),
            context=context,
            aggregators=LLMContextAggregatorPair(context),
            call=call,
        )
    with caplog.at_level(logging.WARNING):
        await open_call(cast(Any, Pipeline()), Greeting(text="Hi."), call, context=context)
    assert ("prompt hooks" in caplog.text) is not flows
