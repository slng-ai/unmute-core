"""OpenAI's realtime and live services in a Pipecat pipeline: lines, refusals, instructions."""

import asyncio
import json
from typing import Any, cast

import pytest
from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.live import events as live_events
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.events import Modality
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from tests.pipecat.harness import until
from tests.pipecat.test_model import Scripted, service
from tests.pipecat.test_telemetry import _pusher
from tests.tasks.job import job, plan, state
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.greeting import Greeting
from unmute_core.hooks import Hooks
from unmute_core.pipecat import (
    IdleNudgeObserver,
    TelemetryObserver,
    UnmuteFlows,
    attach_call,
    attach_realtime,
    greeting_frames,
    open_call,
)
from unmute_core.pipecat._realtime import ask, follow_replies, speak, speaks_itself
from unmute_core.realtime import say_line


def realtime(modalities: list[Modality] | None = None) -> OpenAIRealtimeLLMService:
    """A realtime service that records the client events it would send."""
    llm = OpenAIRealtimeLLMService(
        api_key="test",
        settings=OpenAIRealtimeLLMService.Settings(
            session_properties=events.SessionProperties(output_modalities=modalities),
            system_instruction="You help.",
        ),
    )
    llm.sent = []  # ty: ignore[unresolved-attribute]

    async def send(event: Any) -> None:
        llm.sent.append(event)  # ty: ignore[unresolved-attribute]

    llm.send_client_event = send  # ty: ignore[invalid-assignment]
    return llm


def live() -> OpenAILiveLLMService:
    """A live service that records the client events it would send."""
    llm = OpenAILiveLLMService(api_key="test")
    llm.sent = []  # ty: ignore[unresolved-attribute]

    async def send(event: Any) -> None:
        llm.sent.append(event)  # ty: ignore[unresolved-attribute]

    llm.send_client_event = send  # ty: ignore[invalid-assignment]
    return llm


def sent(llm: Any) -> list[Any]:
    """The client events a recording service sent."""
    return llm.sent


def test_who_speaks_a_fixed_line() -> None:
    assert speaks_itself(realtime())
    assert speaks_itself(realtime(["audio"]))
    assert not speaks_itself(realtime(["text"]))
    assert speaks_itself(live())
    assert not speaks_itself(service(Scripted()))


async def test_a_realtime_line_is_a_reply_on_top_of_the_instructions() -> None:
    llm = realtime()

    await speak(llm, "One moment.")
    await ask(llm, None)

    first, second = sent(llm)
    assert isinstance(first, events.ResponseCreateEvent)
    assert first.response is not None
    assert first.response.instructions == f"You help.\n{say_line('One moment.')}"
    assert second.response.instructions == "You help."


async def test_a_line_waits_for_the_running_reply_to_end() -> None:
    llm = realtime()
    follow_replies(llm)
    await llm.push_frame(LLMFullResponseStartFrame())

    asking = asyncio.create_task(speak(llm, "One moment."))
    await asyncio.sleep(0.05)  # time for a line that does not wait to go out
    assert sent(llm) == []

    await llm.push_frame(LLMFullResponseEndFrame())
    await asyncio.wait_for(asking, 1)
    assert len(sent(llm)) == 1


async def test_a_line_waits_for_its_own_reply_after_a_tool_only_reply() -> None:
    llm = realtime()
    follow_replies(llm)
    context = LLMContext()
    call = FunctionCallFromLLM("end_call", "c1", {}, context)
    await llm.push_frame(FunctionCallsStartedFrame(function_calls=[call]))

    saying = asyncio.create_task(speak(llm, "Goodbye.", wait=True))
    await asyncio.sleep(0.05)  # time for a line that does not wait to go out
    assert sent(llm) == []

    await llm.push_frame(LLMFullResponseEndFrame())
    await until(lambda: sent(llm))
    assert len(sent(llm)) == 1
    assert not saying.done()

    await llm.push_frame(LLMFullResponseStartFrame())
    await llm.push_frame(LLMFullResponseEndFrame())
    await asyncio.wait_for(saying, 1)


async def test_a_live_model_reads_its_lines_as_commentary() -> None:
    llm = live()

    await speak(llm, "One moment.")
    await ask(llm, None)

    [commentary] = sent(llm)
    assert isinstance(commentary, live_events.SessionCommentaryAppendEvent)
    assert commentary.content == "One moment."


def test_the_greeting_on_a_model_that_speaks_itself() -> None:
    fixed = Greeting(text="Hello.")
    [frame] = greeting_frames(fixed, None, llm=realtime())
    assert isinstance(frame, LLMMessagesAppendFrame)
    assert frame.messages == [{"role": "developer", "content": say_line("Hello.")}]
    assert frame.run_llm

    [spoken] = greeting_frames(fixed, None, llm=realtime(["text"]))
    assert isinstance(spoken, TTSSpeakFrame)

    caller_first = Greeting(speaks_first="user")
    assert greeting_frames(caller_first, None, llm=realtime()) == []
    [run] = greeting_frames(caller_first, None, llm=live())
    assert isinstance(run, LLMRunFrame)


def flows(
    llm: Any,
    payload: dict[str, Any] | None = None,
    *,
    save_values: bool = False,
    call: CallHooks | None = None,
) -> UnmuteFlows:
    """An ``UnmuteFlows`` on ``llm``, with no tasks unless ``payload`` has some."""
    task_plan = plan(payload or {"mode": "record", "tasks": []})
    context = LLMContext()
    return UnmuteFlows(
        AgentSpec("You help.", tasks=task_plan, save_values=save_values),
        llm=llm,
        context=context,
        aggregators=LLMContextAggregatorPair(context),
        state=state(task_plan),
        call=call,
    )


def test_unmute_flows_refuses_what_a_realtime_model_cannot_run() -> None:
    with pytest.raises(ValueError, match="tasks do not run on a realtime model"):
        flows(realtime(), payload=job())
    with pytest.raises(ValueError, match="save_values needs instructions that change"):
        flows(live(), save_values=True)
    hooks = Hooks()
    hooks.before_model(lambda request: None)
    task_plan = plan({"mode": "record", "tasks": []})
    with pytest.raises(ValueError, match="the before_model hooks never run"):
        flows(realtime(), call=CallHooks(state(task_plan), hooks=hooks))


async def test_unmute_flows_gives_a_realtime_model_its_instructions_and_values() -> None:
    llm = realtime()
    queued: list[Frame] = []

    async def queue(frame: Frame, *_args: object, **_options: object) -> None:
        queued.append(frame)

    llm.queue_frame = queue  # ty: ignore[invalid-assignment]
    flow = flows(llm, save_values=True)
    flow.install()
    await asyncio.sleep(0)
    flow.state.save({"order_id": "A-12"})
    await asyncio.sleep(0)

    texts = [str(cast(Any, frame).delta.system_instruction) for frame in queued]
    assert len(texts) == 2
    assert texts[0].startswith("You help.\n\nValues of this call.")
    assert "- order_id: A-12" in texts[1]


class Worker:
    """Takes the event handlers ``attach_realtime`` adds."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def add_event_handler(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler

    async def end(self) -> None:
        """End nothing."""


@pytest.mark.parametrize("fixed", [False, True])
async def test_attach_realtime_keeps_the_instructions_current(fixed: bool) -> None:
    llm = live() if fixed else realtime()
    queued: list[Frame] = []

    async def queue(frame: Frame, *_args: object, **_options: object) -> None:
        queued.append(frame)

    llm.queue_frame = queue  # ty: ignore[invalid-assignment]
    task_plan = plan({"mode": "record", "tasks": []})
    call_state = state(task_plan)
    worker = Worker()

    await attach_realtime(llm, worker, "Order {{order_id}}", call_state)  # ty: ignore[invalid-argument-type]
    call_state.save({"order_id": "A-12"})
    await asyncio.sleep(0)

    texts = [frame.delta.system_instruction for frame in queued]  # ty: ignore[unresolved-attribute]
    assert texts[0] == "Order none recorded yet."
    assert texts[1:] == ([] if fixed else ["Order A-12"])


async def test_attach_realtime_runs_the_first_message_tools_and_sends_their_notes() -> None:
    llm = realtime()
    queued: list[Frame] = []

    async def queue(frame: Frame, *_args: object, **_options: object) -> None:
        queued.append(frame)

    llm.queue_frame = queue  # ty: ignore[invalid-assignment]
    hooks = Hooks()
    heard: list[str] = []
    hooks.on("call.first_user_message")(lambda _event, data: heard.append(data["text"]))
    task_plan = plan({"mode": "record", "tasks": []})
    call = CallHooks(state(task_plan), hooks=hooks)
    await attach_realtime(llm, Worker(), "Help.", call=call)  # ty: ignore[invalid-argument-type]

    await llm.push_frame(
        TranscriptionFrame("Where is my order?", "caller", "now"), FrameDirection.UPSTREAM
    )
    await until(lambda: len(queued) == 2)

    assert heard == ["Where is my order?"]
    assert len(queued) == 2


class Lookups(CallHooks):
    """A call whose tools on events save a value as the model is asked."""

    async def before_request(self, caller: str | None) -> None:
        await asyncio.sleep(0.01)
        assert self.state is not None
        self.state.save({"order_id": "A-12"})


class Opening(Worker):
    """A worker that records the frames queued on the pipeline, in order."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.order = order

    async def queue_frames(self, frames: list[Frame]) -> None:
        self.order.extend(type(frame).__name__ for frame in frames)


async def test_a_worded_greeting_on_a_realtime_model_waits_for_the_lookups_and_the_new_instructions() -> (
    None
):
    llm = realtime()
    order: list[str] = []

    async def queue(frame: Frame, *_args: object, **_options: object) -> None:
        order.append(str(cast(Any, frame).delta.system_instruction))

    llm.queue_frame = queue  # ty: ignore[invalid-assignment]
    call = Lookups(state(plan({"mode": "record", "tasks": []})))
    worker = Opening(order)
    await attach_realtime(llm, worker, "Order {{order_id}}", call.state, call=call)  # ty: ignore[invalid-argument-type]

    await open_call(worker, Greeting(), call, context=LLMContext(), llm=llm)  # ty: ignore[invalid-argument-type]

    assert order == ["Order none recorded yet.", "Order A-12", "LLMMessagesAppendFrame"]


async def test_a_fixed_line_on_a_realtime_model_does_not_wait_for_the_lookups() -> None:
    llm = realtime()
    order: list[str] = []

    async def queue(frame: Frame, *_args: object, **_options: object) -> None:
        order.append(str(cast(Any, frame).delta.system_instruction))

    llm.queue_frame = queue  # ty: ignore[invalid-assignment]
    call = Lookups(state(plan({"mode": "record", "tasks": []})))
    worker = Opening(order)
    await attach_realtime(llm, worker, "Order {{order_id}}", call.state, call=call)  # ty: ignore[invalid-argument-type]

    await open_call(worker, Greeting(text="Hello."), call, context=LLMContext(), llm=llm)  # ty: ignore[invalid-argument-type]

    assert order == ["Order none recorded yet.", "LLMMessagesAppendFrame"]


async def test_attach_realtime_refuses_save_values_on_a_live_model() -> None:
    with pytest.raises(ValueError, match="save_values needs instructions that change"):
        await attach_realtime(live(), Worker(), "Help.", save_values=True)  # ty: ignore[invalid-argument-type]


async def test_an_idle_nudge_is_said_by_a_realtime_model() -> None:
    llm = realtime()
    observer = IdleNudgeObserver(None, Worker(), llm=llm)  # ty: ignore[invalid-argument-type]
    observer._played.set()

    async def played() -> None:
        await asyncio.sleep(0.01)
        observer._played.set()

    task = asyncio.create_task(played())
    await observer._speak("Are you still there?")
    await task

    [event] = sent(llm)
    assert event.response.instructions.endswith(say_line("Are you still there?"))


async def test_one_realtime_reply_is_one_llm_span(spans: Any) -> None:
    telemetry, exported = spans
    observer = TelemetryObserver(telemetry, model="gpt-realtime", provider="openai")
    push = _pusher(observer)
    llm = realtime()
    context = LLMContext()
    usage = LLMTokenUsage(prompt_tokens=310, completion_tokens=4, total_tokens=314)

    call = FunctionCallFromLLM("lookup", "c1", {"id": "A-12"}, context)
    await push(FunctionCallsStartedFrame(function_calls=[call]), 1.0, llm)
    await push(MetricsFrame(data=[LLMUsageMetricsData(processor="llm", value=usage)]), 1.1, llm)
    await push(LLMFullResponseEndFrame(), 1.2, llm)
    await push(LLMFullResponseStartFrame(), 2.0, llm)
    await push(LLMFullResponseStartFrame(), 2.1, llm)
    await push(LLMTextFrame("Found it."), 2.2, llm)
    await push(LLMFullResponseEndFrame(), 2.3, llm)

    replies = [dict(span.attributes or {}) for span in exported() if span.name == "llm"]
    assert [(r.get("gen_ai.usage.input_tokens"), r.get("output.value")) for r in replies] == [
        (
            310,
            json.dumps(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"id": "A-12"}'},
                        }
                    ],
                }
            ),
        ),
        (None, "Found it."),
    ]
    assert not any(r.get("llm.interrupted") for r in replies)


def _library_handler() -> Any:
    """The handler of a library tool that keeps running, as ``function_schema`` builds it."""
    from unmute_core.pipecat._tools import function_schema

    async def handler(params: Any) -> None:
        """Do nothing."""

    return function_schema(
        {"name": "slow", "description": "Slow."}, handler, cancel_on_interruption=False
    ).handler


def _with_async_tool(llm: Any) -> str:
    """Register a tool that keeps running through speech, and return the composed instructions."""
    llm.register_function("slow", _library_handler(), cancel_on_interruption=False)
    llm._compose_system_instruction()
    return str(llm.settings.system_instruction)


async def test_attach_realtime_hides_the_async_tool_text_but_keeps_the_tool_async() -> None:
    llm = realtime()
    assert "ASYNC TOOLS" in _with_async_tool(realtime())
    await attach_realtime(llm, Worker(), "Help.")  # ty: ignore[invalid-argument-type]
    assert "ASYNC TOOLS" not in _with_async_tool(llm)
    assert llm._functions["slow"].cancel_on_interruption is False


async def test_attach_call_hides_the_async_tool_text_for_the_llm_it_is_given() -> None:
    llm = service(Scripted())
    attach_call(cast(Any, Worker()), CallHooks(), LLMContext(), llm=llm)
    assert "ASYNC TOOLS" not in _with_async_tool(llm)
    assert "ASYNC TOOLS" in _with_async_tool(service(Scripted()))


def test_unmute_flows_hides_the_async_tool_text_for_its_llm() -> None:
    llm = service(Scripted())
    flows(llm)
    assert "ASYNC TOOLS" not in _with_async_tool(llm)


async def test_waiting_for_the_instructions_gives_up_on_a_resend_that_never_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unmute_core.pipecat import _realtime

    class Service:
        pass

    monkeypatch.setattr(_realtime, "SETTLE_WAIT_SECONDS", 0.05)
    llm = Service()
    never = asyncio.create_task(_forever())
    _realtime.keep_resend(cast(Any, llm), never)
    await asyncio.wait_for(_realtime.settled(cast(Any, llm)), timeout=1)
    never.cancel()


async def _forever() -> None:
    await asyncio.Event().wait()


def _own_async_tool(llm: Any) -> None:
    llm.register_function("own_slow", cast(Any, lambda params: None), cancel_on_interruption=False)


def _composed(llm: Any) -> str:
    llm._compose_system_instruction()
    return str(llm.settings.system_instruction)


def test_an_own_async_tool_keeps_the_async_tool_text_after_attach_call() -> None:
    llm = service(Scripted())
    _own_async_tool(llm)
    attach_call(cast(Any, Worker()), CallHooks(), LLMContext(), llm=llm)
    assert "ASYNC TOOLS" in _composed(llm)


def test_a_library_continue_tool_alone_hides_the_async_tool_text() -> None:
    llm = service(Scripted())
    llm.register_function("transfer", _library_handler(), cancel_on_interruption=False)
    attach_call(cast(Any, Worker()), CallHooks(), LLMContext(), llm=llm)
    assert "ASYNC TOOLS" not in _composed(llm)
    _own_async_tool(llm)
    assert "ASYNC TOOLS" in _composed(llm)


async def test_a_step_tool_that_keeps_running_hides_the_async_tool_text_through_flows() -> None:
    from pipecat.flows import FlowManager

    from tests.pipecat.harness import FakeWorker
    from unmute_core.pipecat._tools import flows_function

    async def handler(args: Any, manager: Any) -> tuple[object, object | None]:
        """Do nothing."""
        return {}, None

    schema = flows_function(
        {"name": "slow", "description": "Slow."}, handler, cancel_on_interruption=False
    )
    flow = FlowManager(llm=cast(Any, None), context_aggregator=None, worker=cast(Any, FakeWorker()))
    wrapped = await flow._create_transition_func("slow", cast(Any, schema.handler))
    llm = service(Scripted())
    llm.register_function("slow", wrapped, cancel_on_interruption=False)
    attach_call(cast(Any, Worker()), CallHooks(), LLMContext(), llm=llm)
    assert "ASYNC TOOLS" not in _composed(llm)
