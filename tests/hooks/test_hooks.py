"""``Hooks``: the order of each point, changing and refusing, observers, class hooks."""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, cast

import pytest

from tests.outputs import AnyResult
from unmute_core.hooks import (
    PROMPT_SUFFIX_MAX,
    EventTool,
    Hooks,
    ModelHandler,
    ModelResponse,
    PromptInfo,
    PromptSuffix,
    Refuse,
    ToolHandler,
    ToolRun,
)
from unmute_core.model import ModelEvent, ModelRequest, TextDelta
from unmute_core.tools import ToolRegistry, local_tool

INFO = PromptInfo("agent", "", "qwen/qwen3-32b")


def _lookup(order_id: str) -> dict:
    """Look an order up."""
    return {"order_id": order_id, "status": "open"}


def _ping() -> dict:
    """Ping."""
    return {"ok": True}


async def test_prompt_and_before_model_run_first_to_last() -> None:
    hooks = Hooks()
    hooks.prompt(lambda text, info: text + " one")

    @hooks.prompt
    def two(text: str, info: PromptInfo) -> str:
        return text + " two"

    hooks.before_model(lambda request: replace(request, params={"a": 1}))
    hooks.before_model(lambda request: None)
    hooks.before_model(lambda request: replace(request, params=request.params | {"b": 2}))

    assert hooks.run_prompt("Help.", INFO) == "Help. one two"

    @hooks.prompt
    async def late(text: str, info: PromptInfo) -> str:
        return text

    with pytest.raises(TypeError, match="must be a plain function"):
        hooks.run_prompt("Help.", INFO)
    request = await hooks.run_before_model(ModelRequest(messages=[]))
    assert request.params == {"a": 1, "b": 2}


async def test_wrap_model_nests_the_first_outermost() -> None:
    order: list[str] = []
    hooks = Hooks()

    def wrap(name: str):
        async def hook(request: ModelRequest, handler: ModelHandler) -> AsyncIterator[ModelEvent]:
            order.append(f"{name} in")
            async for event in handler(request):
                yield event
            order.append(f"{name} out")

        return hook

    hooks.wrap_model(wrap("outer"))
    hooks.wrap_model(wrap("inner"))

    async def send(request: ModelRequest) -> AsyncIterator[ModelEvent]:
        order.append("model")
        yield TextDelta("hi")

    events = [event async for event in hooks.run_model(ModelRequest(messages=[]), send)]
    assert events == [TextDelta("hi")]
    assert order == ["outer in", "inner in", "model", "inner out", "outer out"]


async def test_after_model_runs_last_to_first_and_a_failure_is_only_logged(caplog) -> None:
    order: list[str] = []
    hooks = Hooks()
    hooks.after_model(lambda request, response: order.append("first"))

    @hooks.after_model
    def second(request: ModelRequest, response: ModelResponse) -> None:
        order.append("second")
        raise RuntimeError("broke")

    with caplog.at_level(logging.ERROR):
        await hooks.run_after_model(ModelRequest(messages=[]), ModelResponse("hi"))
    assert order == ["second", "first"]
    assert "an after_model hook failed" in caplog.text


async def test_tool_hooks_change_wrap_and_replace_a_call() -> None:
    order: list[str] = []
    hooks = Hooks()

    @hooks.before_tool
    def fix_id(run: ToolRun) -> ToolRun:
        order.append("before")
        return replace(run, arguments={"order_id": run.arguments["order_id"].upper()})

    @hooks.wrap_tool
    async def timed(run: ToolRun, handler: ToolHandler) -> object:
        order.append("wrap in")
        result = await handler(run)
        order.append("wrap out")
        return result

    hooks.after_tool(lambda run, result: order.append("after last") or None)
    hooks.after_tool(lambda run, result: {**result, "status": "late"})

    registry = ToolRegistry([local_tool(_lookup, output=AnyResult)])
    events: list[tuple[str, dict[str, Any]]] = []
    hooks.on("tool.*")(lambda event, data: events.append((event, data)))
    result = await registry.call("_lookup", {"order_id": "a-1"}, hooks=hooks)

    assert result == {"order_id": "A-1", "status": "late"}
    assert order == ["before", "wrap in", "wrap out", "after last"]
    assert [event for event, _ in events] == ["tool.started", "tool.succeeded"]
    assert events[0][1] == {"name": "_lookup", "source": "model"}


async def test_a_refused_call_never_runs_and_the_model_reads_why() -> None:
    ran: list[str] = []
    hooks = Hooks()
    hooks.before_tool(lambda run: Refuse("Refunds open at 9."))
    hooks.wrap_tool(lambda run, handler: ran.append("wrap"))
    events: list[str] = []
    hooks.on("tool.refused")(lambda event, data: events.append(data["error"]))

    result = await ToolRegistry([local_tool(_lookup, output=AnyResult)]).call(
        "_lookup", {"order_id": "a"}, hooks=hooks
    )
    assert result == {"refused": "Refunds open at 9."}
    assert ran == []
    assert events == ["Refunds open at 9."]


async def test_a_failing_tool_hook_fails_only_that_call() -> None:
    hooks = Hooks()

    @hooks.before_tool
    def broken(run: ToolRun) -> None:
        raise RuntimeError("bug")

    registry = ToolRegistry([local_tool(_lookup, output=AnyResult)])
    with pytest.raises(Exception, match="could not start"):
        await registry.call("_lookup", {"order_id": "a"}, hooks=hooks)


async def test_parallel_calls_share_no_hook_state() -> None:
    hooks = Hooks()

    @hooks.wrap_tool
    async def slow(run: ToolRun, handler: ToolHandler) -> object:
        await asyncio.sleep(0.01)
        return await handler(run)

    registry = ToolRegistry([local_tool(_lookup, output=AnyResult)])
    results = await asyncio.gather(
        *(registry.call("_lookup", {"order_id": str(n)}, hooks=hooks) for n in range(5))
    )
    assert [cast(dict, result)["order_id"] for result in results] == ["0", "1", "2", "3", "4"]


async def test_observers_watch_patterns_and_async_ones_run_as_tasks() -> None:
    seen: list[str] = []
    hooks = Hooks()
    hooks.on("task.*", "group.started")(lambda event, data: seen.append(event))

    @hooks.on("call.ended")
    async def ended(event: str, data: object) -> None:
        seen.append("ended")

    hooks.emit("task.started", {})
    hooks.emit("group.started", {})
    hooks.emit("group.completed", {})
    hooks.emit("call.ended", None)
    await asyncio.sleep(0)
    assert seen == ["task.started", "group.started", "ended"]
    await hooks.notify("call.ended", None)
    assert seen[-1] == "ended"


def test_an_observer_with_no_pattern_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one event"):
        Hooks().on()


async def test_a_class_hooks_in_with_its_methods_and_an_event_tool_is_kept() -> None:
    class Audit:
        events = ("tool.succeeded",)

        def __init__(self) -> None:
            self.seen: list[str] = []

        def before_model(self, request: ModelRequest) -> ModelRequest:
            return replace(request, params={"temperature": 0})

        def on_event(self, event: str, data: dict) -> None:
            self.seen.append(data["name"])

    audit = Audit()
    tool = EventTool(local_tool(_ping, output=AnyResult), ["call_end"])
    hooks = Hooks(audit, tool)
    assert hooks.event_tools == [tool]
    assert (await hooks.run_before_model(ModelRequest(messages=[]))).params == {"temperature": 0}
    hooks.emit("tool.succeeded", {"name": "ping"})
    hooks.emit("tool.failed", {"name": "ping"})
    assert audit.seen == ["ping"]
    with pytest.raises(ValueError, match="hooks in nowhere"):
        hooks.add(object())


async def test_wrap_failure_hooks_nest_and_may_pass_arguments_down() -> None:
    said: list[object] = []
    hooks = Hooks()

    @hooks.wrap_failure
    async def outer(component: str, error: BaseException, handler: Any) -> None:
        said.append(f"outer {component}")
        await handler(line=None)

    @hooks.wrap_failure
    async def inner(component: str, error: BaseException, handler: Any) -> None:
        said.append("inner")
        await handler()

    async def base(line: str | None = "Sorry.") -> None:
        said.append(line)

    await hooks.run_failure("llm", RuntimeError("down"), base)
    assert said == ["outer llm", "inner", None]


async def test_prompt_suffix_follows_unmute() -> None:
    suffix = PromptSuffix("  /no_think \n")
    assert suffix.prompt("Help callers.\n\n", INFO) == "Help callers.\n\n/no_think"
    only_qwen = PromptSuffix("/no_think", model="qwen/qwen3-32b")
    assert only_qwen.prompt("Sum up.", replace(INFO, model="gpt-4.1-mini")) == "Sum up."
    assert only_qwen.prompt("Sum up.", INFO) == "Sum up.\n\n/no_think"
    with pytest.raises(ValueError, match="empty or whitespace"):
        PromptSuffix(" \n")
    with pytest.raises(ValueError, match=f"the bound is {PROMPT_SUFFIX_MAX}"):
        PromptSuffix("x" * 513)
    with pytest.raises(ValueError, match=r"\{\{\.\.\.\}\} placeholder"):
        PromptSuffix("Say {{ name }}")


async def test_an_after_tool_hook_that_raises_keeps_the_tools_result(caplog) -> None:
    hooks = Hooks()

    @hooks.after_tool
    def broken(run: ToolRun, result: object) -> None:
        raise RuntimeError("bug")

    events: list[str] = []
    hooks.on("tool.*")(lambda event, data: events.append(event))
    with caplog.at_level(logging.ERROR):
        result = await ToolRegistry([local_tool(_lookup, output=AnyResult)]).call(
            "_lookup", {"order_id": "a"}, hooks=hooks
        )
    assert result == {"order_id": "a", "status": "open"}
    assert events == ["tool.started", "tool.succeeded"]
    assert "an after_tool hook of _lookup failed" in caplog.text


async def test_arguments_a_wrap_tool_hook_changes_are_checked_again() -> None:
    hooks = Hooks()

    @hooks.wrap_tool
    async def bad(run: ToolRun, handler: ToolHandler) -> object:
        return await handler(replace(run, arguments={"order_id": 7}))

    with pytest.raises(Exception, match="order_id"):
        await ToolRegistry([local_tool(_lookup, output=AnyResult)]).call(
            "_lookup", {"order_id": "a"}, hooks=hooks
        )


async def test_one_calls_hooks_take_in_the_shared_ones() -> None:
    shared = Hooks(EventTool(local_tool(_ping, output=AnyResult), ["call_end"]))
    shared.before_model(lambda request: replace(request, params={"shared": 1}))
    seen: list[str] = []
    shared.on("call.ended")(lambda event, data: seen.append("shared"))

    mine = Hooks(shared)
    mine.before_model(lambda request: replace(request, params=request.params | {"mine": 1}))
    mine.on("call.ended")(lambda event, data: seen.append("mine"))

    request = await mine.run_before_model(ModelRequest(messages=[]))
    assert request.params == {"shared": 1, "mine": 1}
    assert (await shared.run_before_model(ModelRequest(messages=[]))).params == {"shared": 1}
    mine.emit("call.ended", None)
    shared.emit("call.ended", None)
    assert seen == ["shared", "mine", "shared"]
    assert mine.event_tools == shared.event_tools


def test_an_observer_hears_each_event_once_in_the_order_it_was_added() -> None:
    seen: list[str] = []
    hooks = Hooks()
    hooks.on("task.*", "task.started")(lambda event, data: seen.append(f"first {event}"))
    hooks.on("*")(lambda event, data: seen.append(f"all {event}"))
    hooks.emit("task.started", None)
    hooks.emit("not.an.event", None)
    assert seen == ["first task.started", "all task.started", "all not.an.event"]


def test_a_pattern_that_matches_nothing_raises_a_plain_error() -> None:
    with pytest.raises(ValueError, match=r"^'nope\.\*' matches no event\.") as raised:
        Hooks().on("nope.*")
    assert type(raised.value) is ValueError
