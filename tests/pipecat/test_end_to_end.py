"""Text-only calls on a real Pipecat pipeline: a group runs and returns, a handoff, an end.

No transport, no audio and no network. The model is scripted per prompt site.
"""

import asyncio
from collections.abc import AsyncIterator, Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.workers.runner import WorkerRunner

from tests.pipecat.harness import MAIN_PROMPT
from tests.tasks.job import job, plan, registry, specialist_plan, state
from unmute_core.agent import AgentSpec
from unmute_core.model import ModelEvent, ModelRequest, TextDelta, ToolCall
from unmute_core.pipecat import ModelCallLLMService, UnmuteFlows, end_call_tool


class BySite:
    """A ``ModelCall`` that answers from a script per prompt site, in order."""

    def __init__(self, script: dict[str, list[list[ModelEvent]]]) -> None:
        self.script = script
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        for event in self.script[request.site].pop(0):
            yield event


async def _until(check: Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not check():
            await asyncio.sleep(0.01)


async def test_a_two_step_group_runs_end_to_end_on_a_real_pipeline() -> None:
    payload = job()
    task_plan = plan(payload, tools=registry())
    call_state = state(task_plan)
    model = BySite(
        {
            "": [
                [ToolCall("c1", "order_flow", "{}")],
                [TextDelta("Anything else?")],
            ],
            "task:find_order": [
                [TextDelta("What is the order number?")],
                [TextDelta("Thanks."), ToolCall("c2", "finish", '{"order_id": "A-12"}')],
            ],
            "task:take_note": [
                [TextDelta("Any note to add?")],
                [ToolCall("c3", "finish", '{"note": "urgent"}')],
            ],
        }
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)
    llm = ModelCallLLMService(
        model,
        state=call_state,
        settings=OpenAILLMService.Settings(model="scripted", system_instruction=MAIN_PROMPT),
    )
    flows = UnmuteFlows(
        AgentSpec(MAIN_PROMPT, tasks=task_plan),
        llm=llm,
        context=context,
        aggregators=aggregators,
        state=call_state,
    )
    flows.install()
    worker = PipelineWorker(Pipeline([aggregators.user(), llm, aggregators.assistant()]))
    running = asyncio.create_task(WorkerRunner(handle_sigint=False).run(worker))

    async def say(text: str, requests: int) -> None:
        await worker.queue_frame(
            LLMMessagesAppendFrame([{"role": "user", "content": text}], run_llm=True)
        )
        await _until(lambda: len(model.requests) >= requests)

    try:
        await say("I need help with my order.", 2)
        await say("It is A-12.", 4)
        await say("Mark it urgent.", 6)
        await _until(lambda: flows._run is None and len(model.requests) == 6)
    finally:
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(running, 5)

    assert [request.site for request in model.requests] == [
        "",
        "task:find_order",
        "task:find_order",
        "task:take_note",
        "task:take_note",
        "",
    ]
    assert call_state.plain("order_id") == "A-12"
    assert call_state.plain("note") == "urgent"
    last = model.requests[-1]
    assert last.messages[0] == {"role": "system", "content": MAIN_PROMPT}
    assert '"task": "order_flow"' in str(last.messages)
    assert [tool["function"]["name"] for tool in last.tools] == [
        "search_web",
        "normalize",
        "find_order",
        "order_flow",
    ]


async def test_a_handoff_then_an_end_call_on_a_real_pipeline() -> None:
    task_plan = plan(job(), tools=registry())
    call_state = state(task_plan)
    model = BySite(
        {
            "": [
                [TextDelta("Putting you through."), ToolCall("c1", "to_notes", "{}")],
                [TextDelta("Notes desk.")],
                [TextDelta("Goodbye."), ToolCall("c2", "end_call", "{}")],
            ]
        }
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)
    llm = ModelCallLLMService(
        model,
        state=call_state,
        settings=OpenAILLMService.Settings(model="scripted", system_instruction=MAIN_PROMPT),
    )

    ending = end_call_tool("End the call.", "Bye now.", lambda: flows.reply_text)

    async def to_notes(params: FunctionCallParams) -> None:
        notes = AgentSpec("You take notes.", tasks=specialist_plan())
        await flows.hand_off(params, notes, native_tools=[ending])

    flows = UnmuteFlows(
        AgentSpec(MAIN_PROMPT, tasks=task_plan),
        llm=llm,
        context=context,
        aggregators=aggregators,
        state=call_state,
        native_tools=[FunctionSchema("to_notes", "Notes desk.", {}, [], handler=to_notes)],
    )
    flows.install()
    worker = PipelineWorker(Pipeline([aggregators.user(), llm, aggregators.assistant()]))
    running = asyncio.create_task(WorkerRunner(handle_sigint=False).run(worker))
    await worker.queue_frame(
        LLMMessagesAppendFrame([{"role": "user", "content": "Notes please."}], run_llm=True)
    )
    await _until(lambda: len(model.requests) == 2)
    await worker.queue_frame(
        LLMMessagesAppendFrame([{"role": "user", "content": "That's all, bye."}], run_llm=True)
    )
    await asyncio.wait_for(running, 5)

    opening = model.requests[1]
    assert opening.messages[0] == {"role": "system", "content": "You take notes."}
    assert opening.messages[1:] == [
        {"role": "user", "content": "Notes please."},
        {"role": "assistant", "content": "Putting you through."},
    ]
    assert [tool["function"]["name"] for tool in opening.tools] == [
        "search_web",
        "take_note",
        "end_call",
    ]
    assert len(model.requests) == 3
