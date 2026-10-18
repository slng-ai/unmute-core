"""One call on a fake pipeline around a real ``FlowManager``.

The worker records queued frames, ``pump`` applies them to the context as the aggregators
would, and ``calls`` plays one model reply: its text, its tool calls, and their results.
"""

import asyncio
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService

from tests.tasks.job import job, plan, registry, state
from unmute_core.agent import AgentSpec
from unmute_core.call import CallHooks
from unmute_core.pipecat import UnmuteFlows
from unmute_core.pipecat._tasks import result_text
from unmute_core.telemetry import Telemetry
from unmute_core.tools import ToolExecutor

MAIN_PROMPT = "You help callers with their orders."
MISSING = LLMService.MISSING_FUNCTION_CALL_MESSAGE_TEMPLATE


class FakeWorker:
    """What FlowManager and the adapter use of a PipelineWorker."""

    def __init__(self) -> None:
        self.frames: list[Any] = []
        self.log: list[Any] = []
        self.flushes = 0
        self.on_flush: Any = None
        self.ended = False
        self.app_resources: Any = None

    async def queue_frame(self, frame: Any, direction: Any = None) -> None:
        self.frames.append(frame)
        self.log.append(frame)

    async def queue_frames(self, frames: Any, direction: Any = None) -> None:
        for frame in frames:
            await self.queue_frame(frame)

    async def flush_pipeline(self, timeout: float = 5.0) -> bool:
        self.flushes += 1
        self.log.append("flush")
        if self.on_flush is not None:
            self.on_flush()
        return True

    async def end(self, *, reason: str | None = None) -> None:
        self.ended = True
        self.log.append("end")

    def set_reached_downstream_filter(self, types: Any) -> None:
        pass

    def event_handler(self, name: str) -> Any:
        return lambda handler: handler


class FakeLLM:
    """What the adapter uses of an LLM service: its frame hooks."""

    def __init__(self) -> None:
        self.hooks: dict[str, list[Any]] = {}
        self.pushed: list[Any] = []
        self.composed = 0

    def _has_async_tools(self) -> bool:
        return True

    def _compose_system_instruction(self) -> None:
        self.composed += 1

    def add_event_handler(self, name: str, handler: Any) -> None:
        self.hooks.setdefault(name, []).append(handler)

    async def _call_event_handler(self, name: str, *args: Any) -> None:
        """Run each handler of ``name`` in turn, as Pipecat runs a frame hook."""
        for handler in self.hooks.get(name, []):
            await handler(self, *args)

    async def push_frame(self, frame: Any, direction: Any = None) -> None:
        self.pushed.append(frame)


class FakeAggregators:
    def __init__(self, context: LLMContext) -> None:
        self._assistant = SimpleNamespace(has_function_calls_in_progress=False)
        self._user = SimpleNamespace(_context=context)

    def assistant(self) -> Any:
        return self._assistant

    def user(self) -> Any:
        return self._user


class Harness:
    """One call: the main context, the adapter, and a fake pipeline."""

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        mode: str = "record",
        confirm: dict[str, str] | None = None,
        llm: Any = None,
        native_tools: Sequence[FunctionSchema] = (),
        call: CallHooks | None = None,
        then_targets: Mapping[str, Any] | None = None,
        tts: Any = None,
        stt: Any = None,
        summarizer: Any = None,
        task_models: Mapping[str, Any] | None = None,
        telemetry: Telemetry | None = None,
        **executors: ToolExecutor,
    ) -> None:
        payload = payload or job()
        payload["mode"] = mode
        task_plan = plan(payload, confirm=confirm, tools=registry(**executors))
        self.state = state(task_plan)
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.context = LLMContext()
        self.llm: Any = llm or FakeLLM()
        self.worker = FakeWorker()
        self.flows = UnmuteFlows(
            AgentSpec(
                MAIN_PROMPT,
                tasks=task_plan,
                summarizer=summarizer,
                task_models=task_models or {},
            ),
            llm=self.llm,
            context=self.context,
            aggregators=cast(Any, FakeAggregators(self.context)),
            state=self.state,
            native_tools=native_tools,
            call=call,
            then_targets=then_targets,
            tts=tts,
            stt=stt,
            telemetry=telemetry,
        )
        self.tasks = self.flows.tasks
        self.tasks.listen(lambda name, data: self.events.append((name, data)))
        self.flows.install()
        self.system_instruction = self.flows.main_instruction()
        self.runs = 0
        self.spoken: list[TTSSpeakFrame] = []
        self.results: dict[str, list[tuple[Any, Any]]] = {}
        self._ids = itertools.count(1)

    def pump(self) -> list[Any]:
        """Apply queued frames to the context, as the aggregators do."""
        frames, self.worker.frames = self.worker.frames, []
        for frame in frames:
            if isinstance(frame, LLMMessagesAppendFrame):
                self.context.add_messages(frame.messages)
            elif isinstance(frame, LLMMessagesUpdateFrame):
                self.context.set_messages(frame.messages)
            elif isinstance(frame, LLMSetToolsFrame):
                self.context.set_tools(frame.tools)
            elif isinstance(frame, LLMUpdateSettingsFrame):
                self.system_instruction = frame.delta.system_instruction
            elif isinstance(frame, LLMRunFrame):
                self.runs += 1
            elif isinstance(frame, TTSSpeakFrame):
                self.spoken.append(frame)
        return frames

    def user(self, text: str) -> None:
        self.pump()
        self.context.add_message({"role": "user", "content": text})

    def messages(self) -> list[dict[str, Any]]:
        return [cast(dict[str, Any], m) for m in self.context.get_messages() if isinstance(m, dict)]

    def tool_names(self) -> list[str]:
        tools = cast(Any, self.context.tools)
        return [schema.name for schema in tools.standard_tools]

    async def event(self, name: str, frame: Any) -> None:
        """Fire one of the LLM's frame hooks, the way Pipecat does around a frame."""
        await self.llm._call_event_handler(name, frame)

    def handler(self, name: str) -> Any:
        """The handler Pipecat 1.12 registers for ``name``: the one the context advertises."""
        tools = cast(Any, self.context.tools)
        for schema in tools.standard_tools:
            if schema.name == name and schema.handler is not None:
                return schema.handler

        async def missing(params: Any) -> None:
            await params.result_callback(MISSING.format(function_name=name))

        return missing

    async def reply(self, said: str) -> None:
        """A model reply with text and no tool call."""
        self.pump()
        await self.event("on_before_process_frame", LLMContextFrame(self.context))
        await self.event("on_before_push_frame", LLMFullResponseStartFrame())
        await self.event("on_before_push_frame", LLMTextFrame(said))
        await self.event("on_before_push_frame", LLMFullResponseEndFrame())
        self.context.add_message({"role": "assistant", "content": said})
        self.pump()

    async def call(
        self, name: str, arguments: dict[str, Any] | None = None, *, said: str = ""
    ) -> tuple[Any, Any]:
        """The model calling ``name``. Returns the result and its properties."""
        [result] = await self.calls((name, arguments), said=said)
        return result

    async def calls(
        self, *calls: tuple[str, dict[str, Any] | None], said: str = ""
    ) -> list[tuple[Any, Any]]:
        """One reply that says ``said`` and makes ``calls``. Pipecat runs them at once.

        The reply's text reaches the context behind its audio: at a pipeline flush, as a
        result moves to the next node, or after the handlers.
        """
        self.pump()
        await self.event("on_before_process_frame", LLMContextFrame(self.context))
        await self.event("on_before_push_frame", LLMFullResponseStartFrame())
        if said:
            await self.event("on_before_push_frame", LLMTextFrame(said))
        spoken = [said] if said else []

        def land() -> None:
            if spoken:
                self.context.add_message({"role": "assistant", "content": spoken.pop()})

        self.worker.on_flush = land
        ids = [f"call-{next(self._ids)}" for _ in calls]
        self.context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments or {})},
                    }
                    for call_id, (name, arguments) in zip(ids, calls, strict=True)
                ],
            }
        )
        for call_id in ids:
            self.context.add_message(
                {"role": "tool", "tool_call_id": call_id, "content": "IN_PROGRESS"}
            )
            self.results[call_id] = []

        def params_for(name: str, arguments: dict[str, Any] | None, call_id: str) -> Any:
            async def result_callback(
                result: Any, *, properties: FunctionCallResultProperties | None = None
            ) -> None:
                for message in self.context.get_messages():
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "tool"
                        and message.get("tool_call_id") == call_id
                    ):
                        message["content"] = result_text(result)
                self.results[call_id].append((result, properties))
                if properties is not None and properties.on_context_updated:
                    land()
                    await properties.on_context_updated()

            return FunctionCallParams(
                function_name=name,
                tool_call_id=call_id,
                arguments=arguments or {},
                llm=self.llm,
                pipeline_worker=cast(Any, self.worker),
                context=self.context,
                result_callback=result_callback,
            )

        await asyncio.gather(
            *(
                self.handler(name)(params_for(name, arguments, call_id))
                for call_id, (name, arguments) in zip(ids, calls, strict=True)
            )
        )
        await self.event("on_before_push_frame", LLMFullResponseEndFrame())
        self.worker.on_flush = None
        land()
        self.pump()
        for call_id in ids:
            assert len(self.results[call_id]) == 1, self.results
        return [self.results[call_id][0] for call_id in ids]

    def main_reads(self, at: int = -1) -> dict[str, Any]:
        """The run record the main agent reads: a delegate call's reply."""
        records = []
        for message in self.messages():
            if message.get("role") != "tool":
                continue
            try:
                body = json.loads(message["content"])
            except (TypeError, ValueError):
                continue
            if isinstance(body, dict) and "steps" in body and "status" in body:
                records.append(body)
        return records[at]

    def stages(self) -> list[str]:
        return [name for name, _ in self.events]


def runs_llm(properties: Any) -> bool:
    """Whether a result asks for a model reply."""
    return properties is None or properties.run_llm is not False


async def until(done: Callable[[], object]) -> None:
    """Wait until ``done()`` is true, or fail the test after 3 s."""
    async with asyncio.timeout(3):
        while not done():
            await asyncio.sleep(0)
