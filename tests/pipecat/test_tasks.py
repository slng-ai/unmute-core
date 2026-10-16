"""The Pipecat task adapter on a real ``FlowManager``.

Rules that live in the core are tested in tests/tasks. These tests check the Pipecat
side: tools, nodes, frames, the main agent's context, and the order things happen in.
"""

from typing import Any, cast

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.flows import NO_RESPONSE, FlowManager
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMRunFrame,
    LLMUpdateSettingsFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage

from tests.pipecat.harness import (
    MAIN_PROMPT,
    MISSING,
    FakeAggregators,
    FakeLLM,
    FakeWorker,
    Harness,
    runs_llm,
)
from tests.tasks.job import Canned, job, plan, state
from tests.test_handoff import KeyLimited
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.model import STEP_START_CUE
from unmute_core.pipecat import ModelCallLLMService, UnmuteFlows
from unmute_core.pipecat._history import STEP_START_MESSAGE, caller_turns
from unmute_core.pipecat._tasks import ALREADY_HANDLED, for_flows
from unmute_core.tasks._texts import (
    BEGIN_STEP,
    REFUSED_BUSY,
    REFUSED_RERUN,
    SAVE_BEFORE_CALLER,
    finish_before_heard,
)

FOUND = {"found": True, "order": {"id": "A-12"}, "status": "open"}
NOT_FOUND = {"found": False}


async def _in_step(h: Harness, delegate: str = "find_order") -> None:
    h.user("Where is my order A-12?")
    result, properties = await h.call(delegate)
    assert not runs_llm(properties), result


def test_the_main_agent_sees_its_tools_and_one_delegate_per_task_or_group() -> None:
    h = Harness()
    assert h.tool_names() == ["search_web", "normalize", "find_order", "order_flow"]
    assert all(schema.handler for schema in cast(Any, h.context.tools).standard_tools)
    assert h.system_instruction == MAIN_PROMPT


async def test_a_delegate_resolves_flushes_announces_then_opens_its_step() -> None:
    h = Harness()
    h.user("Where is my order?")
    result, properties = await h.call("find_order")
    assert result == {"status": "running the find_order task"}
    assert not runs_llm(properties)
    log = h.worker.log
    assert log.index("flush") < next(
        i for i, frame in enumerate(log) if isinstance(frame, TTSSpeakFrame)
    )
    assert [frame.text for frame in h.spoken] == ["Let me look."]
    assert h.tool_names() == ["lookup_order", "finish"]
    assert h.system_instruction.startswith("Find order none recorded yet.")
    assert h.runs == 1


async def test_a_step_cannot_reach_a_main_tool_it_was_not_given() -> None:
    h = Harness()
    await _in_step(h)
    assert "search_web" not in h.tool_names()
    result, _ = await h.call("search_web")
    assert result == MISSING.format(function_name="search_web")


async def test_a_finish_saves_and_hands_the_main_agent_its_context_back() -> None:
    h = Harness()
    await _in_step(h)
    h.user("It is A-12.")
    result, properties = await h.call("finish", {"order_id": "A-12"}, said="Found it.")
    assert result == {"status": "completed"}
    assert runs_llm(properties)
    assert h.state.plain("order_id") == "A-12"
    assert h.system_instruction == MAIN_PROMPT
    assert h.tool_names() == ["search_web", "normalize", "find_order", "order_flow"]
    record = h.main_reads()
    assert record["status"] == "completed"
    assert record["steps"][0]["said"] == "Found it."
    assert [m["role"] for m in h.messages()] == ["user", "assistant", "tool"]


async def test_a_save_before_the_caller_spoke_in_the_step_is_refused() -> None:
    h = Harness()
    await _in_step(h)
    result, properties = await h.call("finish", {"order_id": "A-12"})
    assert result == {"refused": SAVE_BEFORE_CALLER}
    assert runs_llm(properties)
    assert h.tool_names() == ["lookup_order", "finish"]


async def test_two_delegates_in_one_reply_start_one_run() -> None:
    h = Harness()
    h.user("Help with my order.")
    (first, _), (second, properties) = await h.calls(("find_order", None), ("order_flow", None))
    assert first == {"status": "running the find_order task"}
    assert second == dict(REFUSED_BUSY)
    assert not runs_llm(properties)


async def test_a_call_from_a_step_that_ended_does_nothing() -> None:
    h = Harness()
    await _in_step(h)
    h.user("A-12.")
    flows = h.flows
    run = flows._run
    assert run is not None
    step = run.step
    assert step is not None
    await h.call("finish", {"order_id": "A-12"})
    handler = flows._finish(run, step)
    assert await handler({}, cast(Any, None)) == (dict(ALREADY_HANDLED), NO_RESPONSE)


async def test_every_run_of_a_call_shares_one_flow_manager(monkeypatch: Any) -> None:
    built: list[FlowManager] = []
    initialize = FlowManager.initialize

    async def counted(flow: FlowManager, *args: Any, **kwargs: Any) -> None:
        built.append(flow)
        await initialize(flow, *args, **kwargs)

    monkeypatch.setattr(FlowManager, "initialize", counted)
    h = Harness()
    await _in_step(h)
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    await _in_step(h, "order_flow")
    assert len(built) == 1


@pytest.mark.parametrize("text", ["{{order_id}}", "{{ order.id }}", "\\{{x}}", "{{ 1x }}", "{x}"])
def test_escaped_text_reads_the_same_after_flows_fills_it(text: str) -> None:
    flow = FlowManager(llm=cast(Any, None), context_aggregator=None, worker=cast(Any, FakeWorker()))
    node = flow._render_node("n", {"role_message": for_flows(text), "task_messages": []})
    assert node["role_message"] == text


async def test_a_step_that_cannot_open_hands_the_main_agent_back(monkeypatch: Any) -> None:
    h = Harness()

    async def broken(*_args: Any) -> None:
        raise RuntimeError("no node")

    monkeypatch.setattr(FlowManager, "set_node_from_config", broken)
    h.user("Where is my order?")
    with pytest.raises(RuntimeError):
        await h.call("find_order")
    assert h.system_instruction == MAIN_PROMPT
    assert h.main_reads()["status"] == "failed"
    assert any(isinstance(frame, LLMRunFrame) for frame in h.worker.log)
    assert any(isinstance(frame, LLMUpdateSettingsFrame) for frame in h.worker.log)


async def test_a_main_message_added_during_a_run_comes_back_with_the_main_agent() -> None:
    h = Harness()
    await _in_step(h)
    h.flows.add_main_message({"role": "developer", "content": "The store closes at six."})
    assert "The store closes at six." not in str(h.messages())
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    assert h.messages()[-1] == {"role": "developer", "content": "The store closes at six."}


async def test_a_rerun_before_the_caller_speaks_is_refused_before_any_announce() -> None:
    h = Harness()
    await _in_step(h)
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    h.spoken.clear()
    result, _ = await h.call("find_order")
    assert result == dict(REFUSED_RERUN)
    assert h.spoken == []
    assert "task.refused_rerun" in h.stages()


def _via_job() -> dict[str, Any]:
    payload = job()
    payload["tasks"][0]["tools"] = ["lookup_order", "search_web"]
    payload["tasks"][0]["via"] = ["search_web"]
    return payload


async def test_a_via_tools_success_moves_the_caller_into_its_task() -> None:
    h = Harness(_via_job(), search_web=Canned({"hits": 3}))
    h.user("Search for my order.")
    result, properties = await h.call("search_web", {"q": "order"})
    assert result == {"hits": 3}
    assert not runs_llm(properties)
    assert h.tool_names() == ["lookup_order", "search_web", "finish"]
    names = [call["function"]["name"] for m in h.messages() for call in m.get("tool_calls") or []]
    assert names == ["search_web"]
    assert h.spoken == []
    assert "task.entered_via" in h.stages()


async def test_a_via_tool_that_fails_stays_a_plain_tool() -> None:
    h = Harness(_via_job(), search_web=Canned({"error": "down"}))
    h.user("Search for my order.")
    result, properties = await h.call("search_web", {"q": "order"})
    assert result == {"error": "down"}
    assert runs_llm(properties)
    assert h.system_instruction == MAIN_PROMPT
    assert h.tool_names() == ["search_web", "normalize", "find_order", "order_flow"]


async def test_a_main_call_beside_the_delegate_keeps_its_result_for_the_main_agent() -> None:
    h = Harness(search_web=Canned({"hits": 3}))
    h.user("Find my order and search the web.")
    _, (result, properties) = await h.calls(("find_order", None), ("search_web", None))
    assert result == {"hits": 3}
    assert not runs_llm(properties)
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    replies = [m["content"] for m in h.messages() if m.get("role") == "tool"]
    assert '{"hits": 3}' in replies


async def test_a_finish_with_no_words_after_a_tools_result_is_refused_once() -> None:
    h = Harness(lookup_order=Canned(NOT_FOUND))
    await _in_step(h)
    h.user("Look up A-12.")
    result, _ = await h.call("lookup_order", {"id": "A-12"})
    assert result == NOT_FOUND
    result, _ = await h.call("finish", {})
    assert result == {"refused": finish_before_heard("lookup_order")}
    result, _ = await h.call("finish", {})
    assert result == {"status": "completed"}
    assert h.main_reads()["steps"][0]["not_heard"].startswith("lookup_order returned:")


async def test_a_finish_that_speaks_after_a_tools_result_goes_through() -> None:
    h = Harness(lookup_order=Canned(NOT_FOUND))
    await _in_step(h)
    h.user("Look up A-12.")
    await h.call("lookup_order", {"id": "A-12"})
    result, _ = await h.call("finish", {}, said="I could not find it.")
    assert result == {"status": "completed"}


async def test_a_step_tool_that_returns_after_its_step_ended_asks_for_nothing() -> None:
    h = Harness()
    await _in_step(h)
    run = h.flows._run
    assert run is not None and run.step is not None
    step = run.step
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    handler = h.flows._step_tool(run, step, "lookup_order")
    assert await handler({}, cast(Any, None)) == (dict(ALREADY_HANDLED), NO_RESPONSE)


def _receipt_job() -> dict[str, Any]:
    payload = job()
    payload["tasks"][0]["finish"][0]["say"] = ["Your order is open."]
    return payload


async def test_a_receipt_line_ends_the_run_and_the_main_agent_does_not_reply() -> None:
    h = Harness(_receipt_job(), lookup_order=Canned(FOUND))
    await _in_step(h)
    h.spoken.clear()
    result, properties = await h.call("lookup_order", {"id": "A-12"})
    assert result == {"status": "completed"}
    assert not runs_llm(properties)
    assert [(f.text, f.append_to_context) for f in h.spoken] == [("Your order is open.", True)]


async def test_a_receipt_leaves_the_main_agent_to_answer_a_call_beside_the_delegate() -> None:
    h = Harness(_receipt_job(), lookup_order=Canned(FOUND))
    h.user("Find my order and search.")
    await h.calls(("find_order", None), ("search_web", None))
    h.spoken.clear()
    _, properties = await h.call("lookup_order", {"id": "A-12"})
    assert runs_llm(properties)
    assert {"role": "assistant", "content": "Your order is open."} in h.messages()
    assert [(f.text, f.append_to_context) for f in h.spoken] == [("Your order is open.", False)]


async def test_a_receipt_mid_group_is_spoken_before_the_next_step() -> None:
    h = Harness(_receipt_job(), lookup_order=Canned(FOUND))
    await _in_step(h, "order_flow")
    h.spoken.clear()
    result, _ = await h.call("lookup_order", {"id": "A-12"})
    assert result == {"status": "completed"}
    assert h.spoken[0].text == "Your order is open."
    assert h.tool_names() == ["finish", "go_back_to_step"]


def _provider_job() -> dict[str, Any]:
    payload = job()
    payload["tasks"][1]["when"] = "The caller wants a note taken."
    payload["tasks"][1]["requires"] = ["order_id"]
    return payload


async def test_a_task_that_needs_a_value_runs_its_provider_first_then_resumes() -> None:
    h = Harness(_provider_job(), confirm={"order_id": "find_order"})
    await _in_step(h, "take_note")
    assert h.tool_names() == ["lookup_order", "finish"]
    h.user("A-12.")
    result, _ = await h.call("finish", {"order_id": "A-12"})
    assert result == {"status": "completed"}
    assert h.tool_names() == ["finish"]
    h.user("Urgent.")
    await h.call("finish", {"note": "urgent"})
    assert [row["step"] for row in h.main_reads()["steps"]] == ["find_order", "take_note"]


async def test_a_group_whose_every_step_is_skipped_hands_the_main_agent_its_turn() -> None:
    payload = job()
    payload["tasks"][1]["when"] = "The caller wants a note taken."
    payload["groups"][0]["steps"] = ["find_order"]
    payload["groups"][0]["skip_when_confirmed"] = {"find_order": "order_id"}
    h = Harness(payload, confirm={"order_id": "find_order"})
    h.state.save({"order_id": "A-12"}, step="find_order")
    h.user("Help with my order.")
    _, properties = await h.call("order_flow")
    assert not runs_llm(properties)
    assert h.system_instruction == MAIN_PROMPT
    assert h.runs == 1
    assert h.main_reads()["status"] == "completed"


def _isolated_job() -> dict[str, Any]:
    payload = job()
    payload["groups"][0]["context_scope"] = "isolated"
    return payload


async def test_an_isolated_group_resets_every_step_and_its_cue_is_no_caller_turn() -> None:
    h = Harness(_isolated_job())
    await _in_step(h, "order_flow")
    assert h.messages() == [
        {"role": "assistant", "content": "Let me look."},
        {"role": "developer", "content": STEP_START_CUE},
    ]
    result, _ = await h.call("finish", {"order_id": "A-12"})
    assert result == {"refused": SAVE_BEFORE_CALLER}
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    assert h.messages() == [STEP_START_MESSAGE]
    result, _ = await h.call("finish", {"note": "urgent"})
    assert result == {"refused": SAVE_BEFORE_CALLER}
    await h.call("finish", {"unserved_request": "Cancel it."})
    assert STEP_START_MESSAGE not in h.messages()
    assert [m["content"] for m in h.messages() if m["role"] == "user"] == [
        "Where is my order A-12?"
    ]


async def test_an_isolated_step_entered_again_opens_on_the_callers_newest_turn() -> None:
    h = Harness(_isolated_job())
    await _in_step(h, "order_flow")
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    h.user("No wait, it is A-13.")
    await h.call("go_back_to_step", {"steps": ["find_order"]})
    assert h.messages() == [{"role": "user", "content": "No wait, it is A-13."}]


async def test_a_hand_back_carries_the_turn_it_answers_to_the_main_agent() -> None:
    h = Harness()
    await _in_step(h)
    h.user("Actually, cancel my subscription.")
    await h.call("finish", {"unserved_request": "Cancel the subscription."})
    turns = [(m["role"], m.get("content")) for m in h.messages()]
    assert [role for role, _ in turns[:4]] == ["user", "assistant", "tool", "user"]
    assert turns[3] == ("user", "Actually, cancel my subscription.")
    assert h.main_reads()["status"] == "unserved"


async def test_no_turn_is_carried_when_the_main_agent_already_saw_it() -> None:
    h = Harness()
    await _in_step(h)
    await h.call("finish", {"unserved_request": "Something else."})
    assert [m["content"] for m in h.messages() if m["role"] == "user"] == [
        "Where is my order A-12?"
    ]


async def test_a_status_mode_step_opens_on_unmutes_line() -> None:
    h = Harness(mode="status")
    await _in_step(h, "order_flow")
    assert h.messages()[-1] == {"role": "developer", "content": BEGIN_STEP}
    h.user("A-12.")
    result, _ = await h.call("finish", {"order_id": "A-12"})
    assert result == {"status": "completed"}
    assert h.messages()[-1] == {"role": "developer", "content": BEGIN_STEP}


async def test_a_task_step_span_records_each_step(spans: Any) -> None:
    telemetry, exported = spans
    h = Harness(telemetry=telemetry)
    await _in_step(h, "order_flow")
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    h.user("Urgent.")
    await h.call("finish", {"note": "urgent"})
    steps = [
        (span.attributes["unmute.task.step"], span.attributes["unmute.task.outcome"])
        for span in exported()
        if span.name == "task_step" and span.attributes
    ]
    assert steps == [("find_order", "completed"), ("take_note", "completed")]


async def test_a_provider_message_in_the_context_does_not_break_a_run() -> None:
    """Gemini keeps its thought signatures in the context as ``LLMSpecificMessage``."""
    h = Harness(lookup_order=Canned(FOUND))
    h.user("Where is order A-12?")
    h.context.add_message(LLMSpecificMessage(llm="google", message={"thought_signature": "x"}))
    result, _ = await h.call("find_order")
    assert result == {"status": "running the find_order task"}
    await h.call("lookup_order")
    assert h.state.plain("order_id") == "A-12"
    assert h.main_reads()["status"] == "completed"


@pytest.mark.parametrize(("cut_off", "looked_up"), [(True, False), (False, True)])
async def test_a_confirm_line_cut_off_before_its_end_is_asked_again(
    cut_off: bool, looked_up: bool
) -> None:
    payload = job()
    payload["tasks"][0]["finish"][0]["confirm"] = ["Shall I close order {{input.order_id}}?"]
    lookup = Canned(FOUND)
    h = Harness(payload, lookup_order=lookup)
    await _in_step(h)
    asked, properties = await h.call("lookup_order", {"order_id": "A-12"})
    assert not runs_llm(properties), asked
    assert asked["asked"] == "Shall I close order A-12?"
    await h.event("on_before_process_frame", BotStartedSpeakingFrame())
    ending = InterruptionFrame() if cut_off else BotStoppedSpeakingFrame()
    await h.event("on_before_process_frame", ending)
    h.user("Yes.")
    again, properties = await h.call("lookup_order", {"order_id": "A-12"})
    assert bool(lookup.calls) is looked_up
    assert (isinstance(again, dict) and again.get("asked") == asked["asked"]) is cut_off
    if cut_off:
        assert not runs_llm(properties)


async def test_a_last_n_step_counts_its_entry_turns_on_the_window_it_keeps() -> None:
    """A save waits for one new caller turn in the window the step keeps, not several."""
    payload = job()
    payload["tasks"][1] |= {"history": "last_n", "max_messages": 2}
    h = Harness(payload)
    for text in ("Hi.", "I have an order.", "It is for my sister.", "Change it."):
        h.user(text)
        await h.reply("Okay.")
    await h.call("order_flow")
    h.user("A-12.")
    await h.call("finish", {"order_id": "A-12"})
    assert caller_turns(h.context.get_messages()) < 5
    await h.reply("Any note for the order?")
    assert "go_back_to_step" not in h.tool_names()
    result, _ = await h.call("finish", {"note": "urgent"})
    assert result == {"refused": SAVE_BEFORE_CALLER}
    h.user("Urgent, please.")
    await h.reply("Noted.")
    assert "go_back_to_step" in h.tool_names()
    await h.call("finish", {"note": "urgent"})
    assert h.state.plain("note") == "urgent"


def _native(name: str) -> FunctionSchema:
    return FunctionSchema(name=name, description="A native tool.", properties={}, required=[])


@pytest.mark.parametrize("name", ["search_web", "find_order", "order_flow"])
def test_a_native_tool_named_like_another_tool_of_the_agent_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match=f"the tool '{name}' is offered twice on one agent"):
        Harness(native_tools=[_native(name)])


def _flows(
    payload: dict[str, Any] | None = None,
    *,
    instructions: str = MAIN_PROMPT,
    llm: Any = None,
    call: CallHooks | None = None,
) -> UnmuteFlows:
    """Build a call's flows on the job, the way ``Harness`` does, with other arguments."""
    task_plan = plan(payload or job())
    context = LLMContext()
    return UnmuteFlows(
        AgentSpec(instructions, tasks=task_plan),
        llm=llm or cast(Any, FakeLLM()),
        context=context,
        aggregators=cast(Any, FakeAggregators(context)),
        state=state(task_plan),
        call=call,
    )


def test_a_task_that_reads_a_tool_fails_the_build_when_the_call_has_no_prefetch() -> None:
    payload = job()
    payload["tasks"][1]["inputs"] = [{"kind": "tool", "name": "load_customer"}]
    with pytest.raises(ValueError, match=r"task take_note reads the tool 'load_customer'"):
        _flows(payload, call=CallHooks())


def test_a_task_that_reads_a_tool_builds_without_the_calls_hooks() -> None:
    payload = job()
    payload["tasks"][1]["inputs"] = [{"kind": "tool", "name": "load_customer"}]
    _flows(payload, call=None)


def test_a_placeholder_key_too_long_fails_the_build_only_for_a_model_call_with_a_limit() -> None:
    prompt = "Hi {{" + "a" * 70 + "}}"
    _flows(instructions=prompt)
    with pytest.raises(ValueError, match=r"agent instructions: the placeholder"):
        _flows(instructions=prompt, llm=ModelCallLLMService(cast(Any, KeyLimited("x"))))
