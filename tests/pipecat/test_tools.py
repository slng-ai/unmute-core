"""``tools_schema``: an agent without tasks gets each tool's announce, effect and interruption."""

import asyncio
import json
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSSpeakFrame
from pydantic import BaseModel

from tests.outputs import AnyResult
from tests.pipecat.harness import FakeWorker
from tests.pipecat.test_realtime import live, realtime
from tests.tasks.job import Canned, Sending
from unmute_core.call import CallHooks
from unmute_core.pipecat import tools_schema
from unmute_core.pipecat._tools import ParametersFlowManager, flows_function, tool_dicts
from unmute_core.tools import ToolRegistry, ToolReply, ToolSpec

OPEN = {"type": "object", "properties": {}, "additionalProperties": True}


class Params:
    """The parts of ``FunctionCallParams`` a tool handler uses."""

    def __init__(self) -> None:
        self.arguments: dict[str, Any] = {}
        self.tool_call_id = "call_1"
        self.results: list[tuple[Any, Any]] = []
        self.pushed: list[Any] = []
        self.ended = False
        self.llm = self
        self.pipeline_worker = self

    async def result_callback(self, result: Any, *, properties: Any = None) -> None:
        self.results.append((result, properties))

    async def push_frame(self, frame: Any) -> None:
        self.pushed.append(frame)

    async def end(self) -> None:
        self.ended = True


def _handler(registry: ToolRegistry, name: str) -> Any:
    schema = next(s for s in tools_schema(registry).standard_tools if s.name == name)
    assert schema.handler is not None
    return schema.handler


class Arrival(BaseModel):
    order_id: str
    arrives: date


async def test_a_registry_reply_is_spoken_in_the_call_language_with_no_model_reply() -> None:
    tool = ToolSpec(
        "find",
        "Find.",
        OPEN,
        Arrival,
        Canned({"order_id": "A-12", "arrives": "2026-10-16"}),
    )
    reply = ToolReply(
        tool="find", say=["El pedido {{result.order_id}} llega el {{result.arrives}}."]
    )
    registry = ToolRegistry([tool], replies=[reply])
    schema = next(s for s in tools_schema(registry, language="es").standard_tools)
    assert schema.handler is not None
    params = Speaking()
    await schema.handler(cast(Any, params))
    said = [frame.text for frame in params.pushed if isinstance(frame, TTSSpeakFrame)]
    assert said == ["El pedido A-12 llega el viernes, 16 de octubre de 2026."]
    [(result, properties)] = params.results
    assert result == {"order_id": "A-12", "arrives": "2026-10-16"} and properties.run_llm is False


class Speaking(Params):
    """An LLM service that lets a test report the bot's speech."""

    def __init__(self) -> None:
        super().__init__()
        self.handlers: list[Any] = []

    def add_event_handler(self, name: str, handler: Any) -> None:
        assert name == "on_before_process_frame"
        self.handlers.append(handler)

    async def bot_stopped(self) -> None:
        for handler in self.handlers:
            await handler(self, BotStoppedSpeakingFrame())


async def test_a_tool_that_waits_runs_once_its_line_has_played() -> None:
    executor = Canned({"ok": 1})
    tool = ToolSpec("look", "Look.", OPEN, AnyResult, executor, announce="Hm.", announce_wait=True)
    params = Speaking()
    running = asyncio.create_task(_handler(ToolRegistry([tool]), "look")(cast(Any, params)))
    async with asyncio.timeout(1):
        while not params.pushed:
            await asyncio.sleep(0)
    await asyncio.sleep(0.01)
    assert executor.calls == []
    await params.bot_stopped()
    await running
    assert executor.calls == [{}]
    assert params.results == [({"ok": 1}, None)]


async def test_a_hidden_result_reaches_pipecat_as_ok() -> None:
    tool = ToolSpec("pay", "Pay.", OPEN, AnyResult, Canned({"card": "4111"}), show_result=False)
    params = Params()
    await _handler(ToolRegistry([tool]), "pay")(cast(Any, params))
    assert params.results == [({"ok": True}, None)]


def test_a_tool_that_continues_runs_on_through_the_callers_speech() -> None:
    registry = ToolRegistry(
        [
            ToolSpec("slow", "Slow.", OPEN, AnyResult, Canned(), interruption="continue"),
            ToolSpec("quick", "Quick.", OPEN, AnyResult, Canned()),
        ]
    )
    cancels = {
        name: getattr(_handler(registry, name), "_pipecat_cancel_on_interruption", True)
        for name in ("slow", "quick")
    }
    assert cancels == {"slow": False, "quick": True}


class Together(Params):
    """One call's tool calls: they share the pipeline worker and, with ``llm=``, the service."""

    def __init__(self, worker: Params, call_id: str, llm: Any = None) -> None:
        super().__init__()
        self.tool_call_id = call_id
        self.pipeline_worker = worker
        self.llm = llm if llm is not None else worker


def _ending_and_plain() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                "bye",
                "Bye.",
                OPEN,
                AnyResult,
                Canned({"ok": True}),
                effect="ends_conversation",
            ),
            ToolSpec("look", "Look.", OPEN, AnyResult, Canned({"ok": 1})),
        ]
    )


async def test_a_realtime_service_is_not_sent_the_result_of_a_tool_that_ends_the_call() -> None:
    llm = realtime()
    params = Together(Params(), "call_1", llm)
    await _handler(_ending_and_plain(), "bye")(cast(Any, params))
    assert "call_1" in llm._completed_tool_calls
    assert [result for result, _ in params.results] == [{"ok": True}]
    assert params.pipeline_worker.ended


async def test_a_live_service_is_not_sent_the_result_of_a_tool_that_ends_the_call() -> None:
    llm = live()
    llm._open_function_calls["call_1"] = "response"
    params = Together(Params(), "call_1", llm)
    await _handler(_ending_and_plain(), "bye")(cast(Any, params))
    assert "call_1" not in llm._open_function_calls
    assert [result for result, _ in params.results] == [{"ok": True}]


def _running(llm: Any, **handlers: Any) -> None:
    """Make ``llm`` look like it runs these calls, by id, each with its handler."""
    llm._function_call_tasks = {
        index: SimpleNamespace(tool_call_id=call_id, registry_item=SimpleNamespace(handler=handler))
        for index, (call_id, handler) in enumerate(handlers.items())
    }


def _schema_handlers(registry: ToolRegistry) -> dict[str, Any]:
    tools = tools_schema(registry).standard_tools
    return {schema.name: schema.handler for schema in tools}


async def test_a_realtime_result_waits_for_an_ending_sibling_and_stays_local() -> None:
    llm = realtime()
    registry = _ending_and_plain()
    handlers = _schema_handlers(registry)
    _running(llm, look=handlers["look"], bye=handlers["bye"])
    worker = Params()
    first = Together(worker, "look", llm)
    await handlers["look"](cast(Any, first))
    assert "look" in llm._completed_tool_calls
    assert first.results[0][1].run_llm is False
    await handlers["bye"](cast(Any, Together(worker, "bye", llm)))
    assert {"look", "bye"} <= llm._completed_tool_calls


async def test_a_live_service_holds_no_result_back_beside_an_ending_sibling() -> None:
    from unmute_core.pipecat import _tools

    llm = live()
    handlers = _schema_handlers(_ending_and_plain())
    _running(llm, look=handlers["look"], bye=handlers["bye"])
    first = Together(Params(), "look", llm)
    await handlers["look"](cast(Any, first))
    assert first.results[0][1] is None
    assert llm not in _tools._HELD


async def test_a_held_realtime_result_goes_out_when_the_sibling_does_not_end_the_call() -> None:
    llm = realtime()
    handlers = _schema_handlers(
        ToolRegistry(
            [
                ToolSpec(
                    "bye",
                    "Bye.",
                    OPEN,
                    AnyResult,
                    Canned({"error": "no"}),
                    effect="ends_conversation",
                ),
                ToolSpec("look", "Look.", OPEN, AnyResult, Canned({"ok": 1})),
            ]
        )
    )
    _running(llm, look=handlers["look"], bye=handlers["bye"])
    worker = Params()
    await handlers["look"](cast(Any, Together(worker, "look", llm)))
    assert "look" in llm._completed_tool_calls
    last = Together(worker, "bye", llm)
    await handlers["bye"](cast(Any, last))
    assert "look" not in llm._completed_tool_calls
    assert last.results[0][1] is None
    assert not worker.ended


async def test_plain_tools_have_their_sends_drained_at_the_calls_end() -> None:
    sending = Sending()
    tools = ToolRegistry(
        [ToolSpec("note", "Note.", {"type": "object", "properties": {}}, AnyResult, sending)]
    )
    hooks = CallHooks()
    tools_schema(tools, hooks=hooks)
    await hooks.end("ended")
    assert sending.drained


LOCKED = {
    "type": "object",
    "properties": {"flavor": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["flavor"],
    "additionalProperties": False,
    "description": "Order cupcakes.",
}


def _locked_registry() -> ToolRegistry:
    return ToolRegistry([ToolSpec("order_cupcakes", "Order.", LOCKED, AnyResult, Canned())])


def test_the_openai_adapter_sends_the_registry_schema_as_it_is() -> None:
    registry = _locked_registry()
    sent = OpenAILLMAdapter().to_provider_tools_format(tools_schema(registry))
    assert json.dumps(sent) == json.dumps(registry.schemas())


def test_the_parts_the_other_adapters_read_stay_set() -> None:
    """The Anthropic, Gemini and realtime adapters read ``properties`` and ``required``."""
    (schema,) = tools_schema(_locked_registry()).standard_tools
    assert schema.properties == LOCKED["properties"]
    assert schema.required == ["flavor"]


async def test_a_flows_node_function_sends_the_registry_schema_as_it_is() -> None:
    registry = _locked_registry()

    async def handler(_args: dict[str, Any], _flow: Any) -> tuple[object, None]:
        return {}, None

    manager = ParametersFlowManager(
        llm=cast(Any, None), context_aggregator=None, worker=cast(Any, FakeWorker())
    )
    node_function = flows_function(tool_dicts(registry)[0], handler)
    schema = await manager._create_function_schema(node_function)
    sent = OpenAILLMAdapter().to_provider_tools_format(ToolsSchema(standard_tools=[schema]))
    assert json.dumps(sent) == json.dumps(registry.schemas())
    assert schema.handler is not None
    assert schema.handler is not handler
