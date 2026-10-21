"""The Pipecat driver of a simulated call: ``UnmuteFlows`` on a pipeline with no transport.

The pipeline is the one a call has, with typed text for speech: the user aggregator, the
LLM, ``TextTTS`` (each sentence becomes a moment of silence, so a line is heard as on a
call), an output that plays it, and the assistant aggregator. The caller starts and stops
speaking, then their line goes in as a finished user turn. A turn is over once no model
request or function call is open, no reply is due, and nothing has happened for
``SETTLE_S``.

``python -m unmute_core.testing simulate pipecat`` runs it. See
``unmute_core.testing.simulate``.
"""

import asyncio
import contextlib
import time

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMMessagesAppendFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.workers.runner import WorkerRunner

from unmute_core.call import CALLER_HUNG_UP
from unmute_core.pipecat import ModelCallLLMService, UnmuteFlows, open_call
from unmute_core.pipecat.testing import AgentOutput, TextTTS
from unmute_core.testing._scenario import ConverseSettings
from unmute_core.testing.simulate._agent import AgentCall
from unmute_core.testing.simulate._record import Recorder

SETTLE_S = 0.8
"""How long nothing must happen before the agent's turn is over."""
POLL_S = 0.05
END_WAIT_S = 10.0
"""How long the pipeline may take to finish once the call ends."""


class Output(AgentOutput):
    """``AgentOutput`` that also counts the function calls still running, and notes activity."""

    def __init__(self, recorder: Recorder) -> None:
        """Start with nothing said and no call running."""
        super().__init__(ConverseSettings(audio=False, verbose=False, real_voice=False))
        self.recorder = recorder
        self.running: set[str] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Note each function call's start and result, then do what ``AgentOutput`` does."""
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, FunctionCallInProgressFrame):
                self.running.add(frame.tool_call_id)
            elif isinstance(frame, FunctionCallResultFrame):
                self.running.discard(frame.tool_call_id)
        await super().process_frame(frame, direction)
        if self.last > self.recorder.last_activity:
            self.recorder.touch()


class PipecatCall:
    """One simulated call on Pipecat. See ``unmute_core.testing.simulate._call.TextCall``.

    Attributes:
        ended: True once the agent's side ended the pipeline.
    """

    def __init__(self, call: AgentCall, recorder: Recorder) -> None:
        """Keep the call. Nothing runs until ``start``."""
        self.call = call
        self.recorder = recorder
        self._output = Output(recorder)
        self._worker: PipelineWorker | None = None
        self._flows: UnmuteFlows | None = None
        self._running: asyncio.Task[None] | None = None
        self._leaving = False

    @property
    def ended(self) -> bool:
        """True once the agent's side ended the call: an end frame, or a finished pipeline."""
        if self._leaving:
            return False
        done = self._running is not None and self._running.done()
        return self._output.ended or done

    async def start(self) -> list[str]:
        """Build and run the pipeline, greet, and return what the agent said."""
        spec = self.call.spec
        assert spec.model is not None, "the agent needs its model"
        context = LLMContext()
        aggregators = LLMContextAggregatorPair(context)
        llm = ModelCallLLMService(
            spec.model,
            state=self.call.state,
            settings=OpenAILLMService.Settings(
                model="simulated", system_instruction=spec.instructions
            ),
        )
        flows = UnmuteFlows(
            spec, llm=llm, context=context, aggregators=aggregators, call=self.call.call
        )
        flows.install()
        self._flows = flows
        worker = PipelineWorker(
            Pipeline([aggregators.user(), llm, TextTTS(), self._output, aggregators.assistant()]),
            params=PipelineParams(),
        )
        self._worker = worker
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        self._running = asyncio.create_task(runner.run())
        await open_call(worker, self.call.greeting, self.call.call, context=context)
        return await self._quiet(0)

    async def say(self, line: str) -> list[str]:
        """Type one caller line, as a finished user turn, and return what the agent said."""
        assert self._worker is not None, "start the call first"
        said = len(self._output.said)
        if not self.ended:
            await self._worker.queue_frame(UserStartedSpeakingFrame())
            await asyncio.sleep(0.2)
            await self._worker.queue_frame(UserStoppedSpeakingFrame())
            await self._worker.queue_frame(
                LLMMessagesAppendFrame([{"role": "user", "content": line}], run_llm=True)
            )
            self.recorder.touch()
        return await self._quiet(said)

    async def _quiet(self, said: int) -> list[str]:
        """Wait until the agent's turn is over, and return the lines said since ``said``."""
        output = self._output
        while not self.ended:
            await asyncio.sleep(POLL_S)
            busy = output.answer_due or output.running or self.recorder.open_requests
            if not busy and time.monotonic() - self.recorder.last_activity >= SETTLE_S:
                break
        return output.said[said:]

    async def hang_up(self) -> None:
        """Leave as the caller: make the held saves, end the pipeline, and end the call."""
        self._leaving = True
        self.call.call.caller_left()
        if self._flows is not None:
            await self._flows.settle()
        await self._finish()
        await self.call.call.end(CALLER_HUNG_UP)

    async def _finish(self) -> None:
        """End the pipeline if it still runs, and wait for it."""
        if self._running is None:
            return
        if not self._running.done() and self._worker is not None:
            await self._worker.queue_frame(EndFrame())
        try:
            await asyncio.wait_for(asyncio.shield(self._running), END_WAIT_S)
        except TimeoutError:
            self._running.cancel()

    async def close(self) -> None:
        """End the pipeline and the call if they still run. Never raises."""
        self._leaving = True
        with contextlib.suppress(Exception):
            await self._finish()
        with contextlib.suppress(Exception):
            await self.call.call.end(CALLER_HUNG_UP)
