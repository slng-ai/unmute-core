"""``CallHooks`` runs tools on events, waits before a request, fails a call and ends it."""

import asyncio
import logging
from typing import Any

import pytest
from pydantic import BaseModel

from tests.hooks.helpers import checks, on_end
from tests.outputs import AnyResult
from unmute_core.call import (
    AGENT_ENDED,
    CALLER_HUNG_UP,
    ERROR,
    FAILURE_LINE,
    NOTES_HEADER,
    CallEnd,
    CallHooks,
    EventTool,
)
from unmute_core.hooks import FailureHandler, Hooks, PromptSuffix
from unmute_core.prefetch import Lookup, Prefetch
from unmute_core.state import CallState
from unmute_core.tools import ToolError, ToolRegistry, ToolSpec, Transcript, local_tool

NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}}


class State(BaseModel):
    customer: str | None = None


class Log:
    """What the tools ran with and what the adapter was asked to do."""

    def __init__(self) -> None:
        self.ran: list[tuple[str, dict[str, Any]]] = []
        self.said: list[str] = []
        self.hung_up = 0
        self.transcript: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]

    def tool(self, name: str, result: object = None, *, fail: bool = False, **options: Any):
        """A local tool that logs its call and answers ``result``."""

        async def run() -> object:
            self.ran.append((name, {}))
            if fail:
                raise ToolError(f"{name} broke")
            return result if result is not None else {"ok": True, "tool": name}

        run.__name__ = name
        run.__doc__ = f"The {name} tool."
        return local_tool(run, output=AnyResult, **options)

    async def say(self, line: str) -> None:
        self.said.append(line)

    def hang_up(self) -> None:
        self.hung_up += 1

    def bind(self, call: CallHooks) -> CallHooks:
        call.bind(say=self.say, hang_up=self.hang_up, transcript=lambda: self.transcript)
        return call


async def test_a_background_lookup_runs_after_open_and_the_first_request_waits() -> None:
    log = Log()
    release = asyncio.Event()

    async def hours() -> dict:
        """Read today's hours."""
        await release.wait()
        return {"open": "9:00"}

    state = CallState(State)
    prefetch = Prefetch(
        [
            Lookup(
                "hours", local_tool(hours, output=AnyResult), {"customer": "open"}, background=True
            )
        ]
    )
    call = log.bind(CallHooks(state, prefetch=prefetch))
    assert await call.open()
    assert state.plain("customer") is None
    waiting = asyncio.ensure_future(call.before_request("hello"))
    await asyncio.sleep(0.01)
    assert not waiting.done()
    release.set()
    await waiting
    assert state.plain("customer") == "9:00"
    assert call.results.describe("hours") == '{"open": "9:00"}'
    assert call.notes == ()


async def test_a_lookup_before_the_greeting_runs_inside_open() -> None:
    state = CallState(State)
    prefetch = Prefetch([Lookup("who", lambda _state: {"name": "Ana"}, {"customer": "name"})])
    call = CallHooks(state, prefetch=prefetch)
    assert await call.open()
    assert state.plain("customer") == "Ana"


async def test_a_slow_lookup_stops_holding_the_request_after_the_timeout(caplog) -> None:
    async def stuck() -> dict:
        """Never answers."""
        await asyncio.Event().wait()
        return {}

    state = CallState(State)
    prefetch = Prefetch(
        [Lookup("x", local_tool(stuck, output=AnyResult), {"customer": "x"}, background=True)]
    )
    call = CallHooks(state, prefetch=prefetch, first_request_timeout_s=0.01)
    await call.open()
    with caplog.at_level(logging.WARNING):
        await call.before_request(None)
    assert "still run" in caplog.text


async def test_first_user_message_fires_once_reads_the_message_and_saves() -> None:
    log = Log()
    tool = local_tool(
        _echo(log, "intent"), output=AnyResult, locked={"said": "{{call.first_user_message}}"}
    )
    state = CallState(State)
    call = log.bind(
        CallHooks(
            state,
            hooks=Hooks(EventTool(tool, ["first_user_message"], assign={"customer": "said"})),
        )
    )
    await call.before_request("I want a table")
    await call.before_request("for two")
    assert log.ran == [("intent", {"said": "I want a table"})]
    assert state.plain("customer") == "I want a table"
    assert state.call.first_user_message == "I want a table"
    assert call.notes == ()


def _echo(log: Log, name: str):
    """A tool that logs one ``said`` argument and answers it."""

    async def run(said: str) -> dict:
        log.ran.append((name, {"said": said}))
        return {"ok": True, "said": said}

    run.__name__ = name
    run.__doc__ = f"The {name} tool."
    return run


async def test_tools_follow_a_model_tool_and_each_other_and_note_when_asked() -> None:
    log = Log()
    book = log.tool("book")
    call = log.bind(
        CallHooks(
            hooks=Hooks(
                EventTool(log.tool("notify"), ["tool_succeeded:book"], note=True),
                EventTool(
                    log.tool("audit", show_result=False), ["tool_succeeded:notify"], note=True
                ),
                EventTool(log.tool("apologise"), ["tool_failed:book"], note=True),
            )
        )
    )
    registry = ToolRegistry([book])
    call.watch(registry)
    call.watch(registry)
    await registry.call("book", {}, hooks=call)
    await call.before_request(None)

    assert [name for name, _ in log.ran] == ["book", "notify", "audit"]
    notes = call.notes
    assert len(notes) == 1
    assert notes[0].startswith("notify ran on its own when book succeeded")
    assert call.notes_block() == "\n".join([NOTES_HEADER, *(f"- {note}" for note in notes)])


async def test_a_failed_event_tool_is_noted_even_when_it_hides_results() -> None:
    log = Log()
    book = log.tool("book")
    call = log.bind(
        CallHooks(
            hooks=Hooks(
                EventTool(
                    log.tool("crm", fail=True, show_result=False),
                    ["tool_succeeded:book"],
                    note=True,
                ),
                EventTool(log.tool("retry"), ["tool_failed:crm"], note=True),
            )
        )
    )
    registry = ToolRegistry([book])
    call.watch(registry)
    await registry.call("book", {}, hooks=call)
    await call.before_request(None)
    notes = call.notes
    assert '{"error": "crm broke"}' in notes[0]
    assert notes[1].startswith("retry ran on its own when crm failed")


async def test_a_note_breaks_braces_and_is_capped() -> None:
    log = Log()
    call = log.bind(
        CallHooks(
            hooks=Hooks(
                EventTool(
                    log.tool("big", {"text": "{{x}}" + "a" * 20_000}),
                    ["first_user_message"],
                    note=True,
                )
            )
        )
    )
    await call.before_request("hi")
    note = call.notes[0]
    assert "{{" not in note
    assert len(note) == 15_000


async def test_an_assigned_value_that_does_not_fit_is_only_logged(caplog) -> None:
    log = Log()

    class Counted(BaseModel):
        count: int | None = None

    state = CallState(Counted)
    tool = log.tool("count", {"n": "many"})
    call = CallHooks(
        state, hooks=Hooks(EventTool(tool, ["first_user_message"], assign={"count": "n"}))
    )
    with caplog.at_level(logging.WARNING):
        await call.before_request("hi")
    assert state.plain("count") is None
    assert "saved nothing" in caplog.text


def test_event_tools_refuse_what_they_cannot_run() -> None:
    log = Log()
    with pytest.raises(ValueError, match="no event"):
        EventTool(log.tool("a"), ["call_started"])
    with pytest.raises(ValueError, match="no event"):
        EventTool(log.tool("a"), ["tool_succeeded"])
    with pytest.raises(ValueError, match="work at call start is a prefetch"):
        EventTool(log.tool("a"), ["call_start"])
    with pytest.raises(ValueError, match="drop the announce"):
        EventTool(log.tool("a", announce="One moment."), ["call_end"])
    with pytest.raises(ValueError, match="argument said has no source"):
        EventTool(local_tool(_echo(log, "a"), output=AnyResult), ["call_end"])
    with pytest.raises(ValueError, match="assigns 'nope'"):
        CallHooks(
            CallState(State), hooks=Hooks(EventTool(log.tool("a"), ["call_end"], {"nope": "x"}))
        )
    with pytest.raises(ValueError, match="needs state"):
        CallHooks(prefetch=Prefetch([]))
    with pytest.raises(ValueError, match="loop: a -> b -> a"):
        CallHooks(
            hooks=Hooks(
                EventTool(log.tool("a"), ["tool_succeeded:b"]),
                EventTool(log.tool("b"), ["tool_succeeded:a"]),
            )
        )
    chain = [EventTool(log.tool("t0"), ["call_end"])] + [
        EventTool(log.tool(f"t{i}"), [f"tool_succeeded:t{i - 1}"]) for i in range(1, 9)
    ]
    with pytest.raises(ValueError, match="deeper than 8"):
        CallHooks(hooks=Hooks(*chain))
    CallHooks(hooks=Hooks(*chain[:8]))


async def test_a_failed_check_before_the_greeting_fails_the_call() -> None:
    log = Log()

    async def broken() -> None:
        raise RuntimeError("tts is down")

    state = CallState(State)
    prefetch = Prefetch([Lookup("x", log.tool("x"), {"customer": "tool"}, background=True)])
    call = log.bind(CallHooks(state, prefetch=prefetch, hooks=checks(broken)))
    assert not await call.open()
    assert log.said == [FAILURE_LINE]
    assert log.hung_up == 1
    assert log.ran == []
    assert call.failed


async def test_a_slow_check_times_out() -> None:
    log = Log()
    call = log.bind(CallHooks(hooks=checks(asyncio.Event().wait), ready_timeout_s=0.01))
    assert not await call.open()
    assert log.hung_up == 1


async def test_a_tts_failure_only_logs_and_the_call_stays_open(caplog) -> None:
    played: list[str] = []
    hooks = Hooks()

    @hooks.wrap_failure
    async def audio(component: str, error: BaseException, handler: FailureHandler) -> None:
        played.append(component)
        await handler(line=None)

    heard: list[str] = []
    hooks.on("call.tts_failed")(lambda event, data: heard.append(data["component"]))
    log = Log()
    call = log.bind(CallHooks(hooks=hooks))
    with caplog.at_level("WARNING"):
        await call.fail("tts", RuntimeError("voice down"))
    assert "voice down" in caplog.text
    assert heard == ["tts"]
    assert played == []
    assert log.said == []
    assert log.hung_up == 0
    assert not call.failed

    await call.fail("llm", RuntimeError("down"))
    assert played == ["llm"]
    assert log.hung_up == 1


async def test_a_failure_hook_that_never_hangs_up_keeps_the_call_going() -> None:
    log = Log()
    seen: list[str] = []

    hooks = Hooks()

    @hooks.wrap_failure
    async def hook(component: str, error: BaseException, handler: FailureHandler) -> None:
        seen.append(component)

    hooks.on("call.recovered")(lambda event, data: seen.append(f"recovered {data['component']}"))
    call = log.bind(CallHooks(hooks=hooks))
    await call.fail("stt", RuntimeError("down"))
    assert seen == ["stt", "recovered stt"]
    assert log.said == []
    assert log.hung_up == 0
    assert not call.failed

    await call.fail("llm", RuntimeError("down too"))
    assert seen[2:] == ["llm", "recovered llm"]
    assert (await call.end("ended")).reason == "ended"


async def test_the_call_hangs_up_when_the_line_cannot_play() -> None:
    log = Log()

    async def say(line: str) -> None:
        raise RuntimeError("no voice")

    call = CallHooks()
    call.bind(say=say, hang_up=log.hang_up, transcript=list)
    await call.fail("llm", RuntimeError("down"))
    assert log.hung_up == 1


async def test_the_end_settles_runs_the_end_tools_tells_the_hook_and_drains() -> None:
    log = Log()
    state = CallState(State)
    state.save({"customer": "Ana"})
    ended: list[CallEnd] = []
    order: list[str] = []
    report = local_tool(
        _transcript_echo(log, order), output=AnyResult, locked={"transcript": Transcript(text=True)}
    )
    call = log.bind(
        CallHooks(
            state,
            hooks=on_end(
                lambda end: (order.append("hook"), ended.append(end))[0],
                EventTool(report, ["call_end"]),
            ),
        )
    )
    first, second = await asyncio.gather(call.end("participant_disconnected"), call.end("x"))

    assert first is second
    assert order == ["report", "hook"]
    assert log.ran == [("report", {"transcript": "user: hi"})]
    assert ended[0].reason == "participant_disconnected"
    assert ended[0].values == {"customer": "Ana"}
    assert ended[0].transcript == log.transcript
    assert call.notes == ()


def _transcript_echo(log: Log, order: list[str]):
    """A call-end tool that logs the transcript it got."""

    async def report(transcript: str) -> dict:
        """Report the call."""
        order.append("report")
        log.ran.append(("report", {"transcript": transcript}))
        return {"ok": True}

    return report


async def test_the_end_reason_is_error_after_a_failure_and_a_hook_error_is_logged(
    caplog,
) -> None:
    log = Log()

    def hook(end: CallEnd) -> None:
        raise RuntimeError("hook broke")

    call = log.bind(CallHooks(hooks=on_end(hook)))
    await call.fail("llm", RuntimeError("down"))
    with caplog.at_level(logging.ERROR):
        ended = await call.end("user_initiated")
    assert ended.reason == "error"
    assert "an observer of call.ended failed" in caplog.text


async def test_the_end_is_bounded() -> None:
    async def stuck(end: CallEnd) -> None:
        await asyncio.Event().wait()

    call = CallHooks(hooks=on_end(stuck), end_timeout_s=0.01)
    ended = await call.end("user_initiated")
    assert ended.reason == "user_initiated"


async def test_the_end_drains_webhooks_that_send_in_the_background() -> None:
    drained: list[str] = []

    class Background:
        async def run(self, arguments: dict, state: object) -> object:
            return {"ok": True}

        async def drain(self) -> None:
            drained.append("x")

    registry = ToolRegistry([ToolSpec("ping", "Ping.", NO_ARGS, AnyResult, Background())])
    call = CallHooks()
    call.watch(registry)
    await call.end("user_initiated")
    assert drained == ["x"]


async def test_the_hook_runs_with_the_values_when_a_call_end_tool_is_too_slow() -> None:
    async def slow() -> dict:
        """Never answers in time."""
        await asyncio.Event().wait()
        return {}

    state = CallState(State)
    state.save({"customer": "Ana"})
    ended: list[CallEnd] = []
    call = CallHooks(
        state,
        hooks=on_end(ended.append, EventTool(local_tool(slow, output=AnyResult), ["call_end"])),
        end_timeout_s=0.01,
    )
    await call.end("participant_disconnected")
    assert ended[0].values == {"customer": "Ana"}


async def test_a_failure_before_bind_plays_once_the_adapter_binds() -> None:
    log = Log()
    call = CallHooks()
    await call.fail("llm", RuntimeError("down"))
    assert log.said == []
    log.bind(call)
    await asyncio.sleep(0.01)
    assert log.said == [FAILURE_LINE]
    assert log.hung_up == 1


async def test_a_failure_after_the_end_started_changes_nothing() -> None:
    log = Log()
    call = log.bind(CallHooks())
    ended = await call.end("participant_disconnected")
    await call.fail("llm", RuntimeError("late"))
    assert not call.failed
    assert log.said == []
    assert ended.reason == "participant_disconnected"


async def test_the_end_stops_a_background_lookup_and_still_runs_the_call_end_tools() -> None:
    log = Log()

    async def slow() -> dict:
        """Takes longer than the call."""
        await asyncio.sleep(10)
        return {"open": "late"}

    state = CallState(State)
    prefetch = Prefetch(
        [Lookup("x", local_tool(slow, output=AnyResult), {"customer": "open"}, background=True)]
    )
    call = log.bind(
        CallHooks(
            state, prefetch=prefetch, hooks=Hooks(EventTool(log.tool("report"), ["call_end"]))
        )
    )
    await call.open()
    async with asyncio.timeout(1):
        await call.end("participant_disconnected")
    await asyncio.sleep(0.01)
    assert [name for name, _ in log.ran] == ["report"]
    assert state.plain("customer") is None


async def test_a_failure_stops_a_background_lookup() -> None:
    ran: list[str] = []

    async def slow() -> dict:
        """Takes long."""
        await asyncio.sleep(0.05)
        ran.append("done")
        return {"open": "late"}

    state = CallState(State)
    prefetch = Prefetch(
        [Lookup("x", local_tool(slow, output=AnyResult), {"customer": "open"}, background=True)]
    )
    call = Log().bind(CallHooks(state, prefetch=prefetch))
    await call.open()
    await call.fail("llm", RuntimeError("down"))
    await asyncio.sleep(0.1)
    assert ran == []


async def test_a_failed_check_runs_no_lookup_before_the_greeting() -> None:
    ran: list[str] = []

    def who(_state: CallState) -> dict:
        ran.append("who")
        return {"name": "Ana"}

    async def broken() -> None:
        raise RuntimeError("down")

    state = CallState(State)
    prefetch = Prefetch([Lookup("who", who, {"customer": "name"})])
    call = Log().bind(CallHooks(state, prefetch=prefetch, hooks=checks(broken)))
    assert not await call.open()
    assert ran == []


async def test_the_end_drains_a_prefetch_tool() -> None:
    drained: list[str] = []

    class Background:
        async def run(self, arguments: dict, state: object) -> object:
            return {"ok": True}

        async def drain(self) -> None:
            drained.append("x")

    tool = ToolSpec("ping", "Ping.", NO_ARGS, AnyResult, Background())
    call = CallHooks(CallState(State), prefetch=Prefetch([Lookup("ping", tool, {})]))
    await call.end("user_initiated")
    assert drained == ["x"]


async def test_a_failure_hook_that_raises_after_the_line_hangs_up_once() -> None:
    log = Log()
    hooks = Hooks()

    @hooks.wrap_failure
    async def noisy(component: str, error: BaseException, handler: FailureHandler) -> None:
        await handler()
        raise RuntimeError("pager down")

    call = log.bind(CallHooks(hooks=hooks))
    await call.fail("llm", RuntimeError("down"))
    assert log.said == [FAILURE_LINE]
    assert log.hung_up == 1


async def test_two_calls_at_once_share_hooks_and_tools_and_nothing_else() -> None:
    shared_log = Log()
    shared = Hooks(EventTool(shared_log.tool("notify"), ["tool_succeeded:book"], note=True))

    @shared.wrap_tool
    async def slow(run: Any, handler: Any) -> object:
        await asyncio.sleep(0.01)
        return await handler(run)

    class Seen:
        """One call's own observer, built per call."""

        events = ("tool.succeeded",)

        def __init__(self) -> None:
            self.names: list[str] = []

        def on_event(self, event: str, data: dict) -> None:
            self.names.append(data["name"])

    registry = ToolRegistry([shared_log.tool("book")])
    calls = []
    for _ in range(2):
        seen = Seen()
        call = Log().bind(CallHooks(CallState(State), hooks=Hooks(shared, seen)))
        calls.append((call, seen))

    async def run(call: CallHooks, books: int, caller: str) -> None:
        await call.before_request(caller)
        for _ in range(books):
            await registry.call("book", {}, hooks=call)
        await call.before_request(None)

    (first, first_seen), (second, second_seen) = calls
    await asyncio.gather(run(first, 2, "hi from one"), run(second, 1, "hi from two"))

    assert len(first.notes) == 2
    assert len(second.notes) == 1
    assert sorted(first_seen.names) == ["book", "book", "notify", "notify"]
    assert sorted(second_seen.names) == ["book", "notify"]
    assert shared.has("wrap_tool")
    with pytest.raises(ValueError, match="one CallHooks per call"):
        Log().bind(first)


async def test_a_failure_after_an_end_was_requested_is_ignored() -> None:
    log = Log()
    call = log.bind(CallHooks())
    call.end_requested(AGENT_ENDED)
    await call.fail("llm", RuntimeError("late"))
    assert not call.failed
    assert log.said == []
    assert log.hung_up == 0
    assert (await call.end("ended")).reason == AGENT_ENDED


async def test_the_first_requested_end_is_kept() -> None:
    call = Log().bind(CallHooks())
    call.end_requested(AGENT_ENDED)
    call.caller_left()
    assert (await call.end("ended")).reason == AGENT_ENDED


async def test_a_failure_before_the_request_still_ends_with_error() -> None:
    log = Log()
    call = log.bind(CallHooks())
    await call.fail("llm", RuntimeError("down"))
    call.end_requested(AGENT_ENDED)
    assert log.said == [FAILURE_LINE]
    assert (await call.end("ended")).reason == ERROR


async def test_the_caller_leaving_ends_with_caller_hung_up() -> None:
    call = Log().bind(CallHooks())
    call.caller_left()
    assert (await call.end("ended")).reason == CALLER_HUNG_UP


async def test_an_event_tool_note_keeps_the_result_instructions_before_the_result() -> None:
    log = Log()
    call = log.bind(
        CallHooks(
            hooks=Hooks(
                EventTool(
                    log.tool("notify", result_instructions="Read it aloud."),
                    ["tool_succeeded:book"],
                    note=True,
                ),
                EventTool(
                    log.tool("retry", fail=True, result_instructions="Never shown."),
                    ["tool_succeeded:book"],
                    note=True,
                ),
            )
        )
    )
    registry = ToolRegistry([log.tool("book")])
    call.watch(registry)
    await registry.call("book", {}, hooks=call)
    await call.before_request(None)

    assert call.notes[0].startswith(
        "notify ran on its own when book succeeded. Use the result like this: Read it aloud. "
        "Result: "
    )
    assert "Use the result" not in call.notes[1]


async def test_an_event_listed_twice_runs_its_tool_once_and_failure_reaches_observers() -> None:
    log = Log()
    hooks = Hooks(EventTool(log.tool("audit"), ["call_end", "call_end"]))
    heard: list[tuple[str, dict[str, Any]]] = []
    hooks.on("call.failed")(lambda event, data: heard.append((event, data)))
    call = log.bind(CallHooks(hooks=hooks))
    error = RuntimeError("stt down")
    await call.fail("stt", error)
    assert heard[0][1]["component"] == "stt"
    assert heard[0][1]["error"] is error
    await call.end("hang_up")
    assert [name for name, _ in log.ran] == ["audit"]


def test_a_tool_that_assigns_a_missing_value_names_the_values_the_state_has() -> None:
    log = Log()
    hooks = Hooks(EventTool(log.tool("audit"), ["call_end"], assign={"nope": "x"}))
    with pytest.raises(ValueError, match="The state has: customer"):
        CallHooks(CallState(State), hooks=hooks)


def test_skipped_model_hooks_warn_once_and_never_when_an_adapter_runs_them(caplog) -> None:
    hooks = Hooks()
    hooks.add(PromptSuffix("Be brief."))
    call = CallHooks(hooks=hooks)
    assert call.model_hooks == ["prompt"]
    with caplog.at_level("WARNING"):
        call.warn_skipped_model_hooks("fix it")
        call.warn_skipped_model_hooks("fix it")
    assert caplog.text.count("fix it") == 1

    served = CallHooks(hooks=hooks)
    served.note_model_requests()
    with caplog.at_level("WARNING"):
        served.warn_skipped_model_hooks("served")
    assert "served" not in caplog.text
    assert CallHooks().model_hooks == []
