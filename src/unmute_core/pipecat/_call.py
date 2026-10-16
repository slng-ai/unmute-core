"""Opening a Pipecat call: its facts, its greeting, its limit, and its ``CallHooks``.

Ported from unmute@868ae23 templates/pipecat_v1/utils_telephony.py.tmpl:11-36 and
call.py.tmpl:384-395, 628-645.
"""

import asyncio
import time
from collections.abc import Mapping
from typing import Any, cast

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMUpdateSettingsFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.runner.types import CallData
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy

from unmute_core.call import ENDED, MAX_DURATION, NOTES_HEADER, CallHooks, transcript_messages
from unmute_core.greeting import GREETING_HOLD_SECONDS, Greeting
from unmute_core.handoff import newest_caller
from unmute_core.idle import CallLimit
from unmute_core.model import render_prompt
from unmute_core.pipecat._model import ModelCallLLMService
from unmute_core.pipecat._realtime import (
    fixed_instructions,
    follow_replies,
    is_realtime,
    keep_resend,
    settled,
    speak,
    speaks_itself,
    wait_for_caller,
)
from unmute_core.pipecat._requests import RequestHooks
from unmute_core.pipecat._services import hide_async_guidance
from unmute_core.realtime import say_line
from unmute_core.state import CallInfo, CallState, call_info
from unmute_core.tools import values_block

SKIPPED_MODEL_HOOKS = (
    "pass llm= to attach_call or open_call, or give the call to UnmuteFlows(call=)"
)
"""How to make the call's model hooks run, for the warning when nothing runs them."""


def telephony_call_info(
    call_data: CallData | Mapping[str, Any],
    *,
    trunk_number: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
) -> CallInfo:
    """Return what a phone call carries, from the runner's parsed carrier handshake.

    It reads Twilio, Telnyx, Plivo and Exotel handshakes. ``carrier_call_id`` is the
    handshake's ``call_id``. ``direction``, ``from_number`` and ``to_number`` come from its
    ``body``, the parameters your TwiML sends, else from the handshake's own ``from`` and
    ``to``, as Telnyx and Exotel send them. A call without a direction is inbound.

    Args:
        call_data: ``runner_args.call_data`` as the Pipecat runner parsed it.
        trunk_number: The agent's own number, when the handshake does not name it.
        agent_id: The agent's id, the same on every framework. ``UNMUTE_AGENT_ID`` when None.
        session_id: The call's id, when your dispatch gives one. A fresh one otherwise.
    """
    body: Mapping[str, Any] = call_data.get("body") or {}
    direction = str(body.get("direction") or "inbound")
    outbound = direction == "outbound"
    caller = body.get("from_number") or call_data.get("from")
    called = body.get("to_number") or call_data.get("to")
    remote = called if outbound else caller
    local = (caller if outbound else called) or trunk_number
    return call_info(
        direction,  # ty: ignore[invalid-argument-type]
        remote_number=remote,
        local_number=local,
        carrier_call_id=call_data.get("call_id"),
        agent_id=agent_id,
        session_id=session_id,
    )


def greeting_frames(
    greeting: Greeting, state: CallState | None, *, llm: LLMService[Any] | None = None
) -> list[Frame]:
    """Return the frames that open the call: its line, a request to word one, or none.

    Queue them once the pipeline runs and prefetch has filled the state, like on
    ``on_client_connected``. A worded greeting joins the context as a developer message, as
    an idle nudge does: Pipecat drops a first system message when the service has its own
    instructions, and no longer takes the system prompt in the context from Pipecat 2.0.

    Args:
        greeting: How the call opens.
        state: The call state the line's values come from.
        llm: The call's LLM service. OpenAI's realtime or live service that speaks
            itself is told to say a fixed line, in the first request that reaches it. A
            live service opens its session on a first request, so it gets one when the
            caller speaks first, as in Unmute.
    """
    if greeting.speaks_first == "user":
        return [LLMRunFrame()] if llm is not None and fixed_instructions(llm) else []
    line = greeting.line(state)
    if line is not None and not (llm is not None and speaks_itself(llm)):
        return [TTSSpeakFrame(line)]
    instruction = greeting.instruction if line is None else say_line(line)
    message: Any = {"role": "developer", "content": instruction}
    return [LLMMessagesAppendFrame(messages=[message], run_llm=True)]


def max_duration(
    worker: PipelineWorker, seconds: float, *, call: CallHooks | None = None
) -> CallLimit:
    """End the call ``seconds`` after now, with no closing line. Call it once it has opened.

    The count stops when the pipeline ends first.

    Args:
        worker: The call's pipeline worker.
        seconds: The longest the call may run.
        call: The call's hooks. The end is noted as ``MAX_DURATION``, so the call ends with
            that reason.

    Returns:
        The running limit.
    """

    async def end() -> None:
        """Note the end, then end the pipeline."""
        if call is not None:
            call.end_requested(MAX_DURATION)
        await worker.queue_frame(EndFrame())

    limit = CallLimit(seconds, end)

    async def finished(_worker: object, _frame: object) -> None:
        """Stop counting: the call ended first."""
        limit.cancel()

    worker.add_event_handler("on_pipeline_finished", finished)
    limit.start()
    return limit


class GreetingHold(BaseUserMuteStrategy):
    """Holds the caller's audio until the greeting starts, or until it ends.

    Give it to the user aggregator: ``LLMUserAggregatorParams(user_mute_strategies=[hold])``.
    Pipecat's ``MuteUntilFirstBotCompleteUserMuteStrategy`` does the same for a greeting
    nobody may cut off, but it never lets go when the caller speaks first. A greeting that
    never starts or never ends would hold the caller forever, so the hold lets go after
    ``max_hold_s``, counted from ``arm``.
    """

    def __init__(
        self, *, interruptible: bool = True, max_hold_s: float = GREETING_HOLD_SECONDS
    ) -> None:
        """Start holding.

        Args:
            interruptible: True lets go as the greeting starts, so the caller can cut in.
                False lets go once it has played. ``open_call`` sets it from the greeting.
            max_hold_s: The longest the hold lasts once ``arm`` started the clock. The
                clock never runs before: the call's checks and prefetch do not count.
        """
        super().__init__()
        self.interruptible = interruptible
        self.max_hold_s = max_hold_s
        self._held = True
        self._started = False
        self._deadline: float | None = None

    def arm(self) -> None:
        """Start the clock. ``open_call`` calls it once the greeting is queued."""
        self._deadline = time.monotonic() + self.max_hold_s

    def release(self) -> None:
        """Let go now, like when the caller speaks first."""
        self._held = False

    async def process_frame(self, frame: Frame) -> bool:
        """Return True while the caller's audio is held."""
        await super().process_frame(frame)
        if self._held and self._deadline is not None and time.monotonic() >= self._deadline:
            self._held = False
        if not self._held:
            return False
        if isinstance(frame, BotStartedSpeakingFrame):
            self._started = True
            self._held = not self.interruptible
        elif (isinstance(frame, BotStoppedSpeakingFrame) and self._started) or (
            isinstance(frame, ErrorFrame) and not self._started
        ):
            self._held = False
        return self._held


async def fill_prompt(
    llm: LLMService[Any],
    worker: PipelineWorker,
    prompt: str,
    state: CallState,
    *,
    save_values: bool = False,
) -> None:
    """Keep a plain pipeline's prompt filled from the call state, for a cascade LLM service.

    Await it before the pipeline runs, for a pipeline without ``UnmuteFlows``. It sends the
    filled prompt as the service's system instruction now, and again after each save, so a
    value saved by prefetch or a tool reaches the next request. The call's ``prompt`` hooks
    still run through ``attach_call(llm=)``.

    Args:
        llm: The call's LLM service.
        worker: The call's worker. Its end stops following the state.
        prompt: The agent's prompt, placeholders still in it.
        state: The call state the placeholders read.
        save_values: End the prompt with the current values, for ``values_tool``.

    Raises:
        TypeError: If the service is a realtime one: use ``attach_realtime``.
    """
    if is_realtime(llm):
        raise TypeError(f"{type(llm).__name__} is a realtime service: use attach_realtime")

    def instruct() -> Frame:
        """Return the frame that sends the filled prompt now."""
        text = render_prompt(prompt, state)
        if save_values and state.names:
            text = f"{text}\n\n{values_block(state)}"
        return LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=text))

    await llm.queue_frame(instruct())
    unlisten = state.listen(
        lambda _event: keep_resend(llm, asyncio.create_task(llm.queue_frame(instruct())))
    )

    async def finished(_worker: object, _frame: object) -> None:
        """Stop following the state."""
        unlisten()

    worker.add_event_handler("on_pipeline_finished", finished)


def attach_call(
    worker: PipelineWorker,
    call: CallHooks,
    context: LLMContext,
    *,
    llm: LLMService[Any] | None = None,
    transport: BaseTransport | None = None,
) -> None:
    """Let ``call`` speak and hang up on the pipeline, fail it and end it. Once per call.

    An error that leaves a service unable to work fails the call: the line plays, then the
    call ends. Keep the worker's ``processor_unusable_policy`` at ``CONTINUE`` so the line
    can play. The end of the pipeline ends the call.

    Args:
        worker: The call's pipeline worker.
        call: The call's hooks.
        context: The shared context, for the transcript and the notes.
        llm: The LLM service of a pipeline without ``UnmuteFlows``. Each request then
            waits for the tools on events, ends with their notes, and runs the call's
            ``prompt`` and model hooks. A failed ``ModelCallLLMService`` request fails the
            call. ``UnmuteFlows(call=)`` does this itself. OpenAI's realtime or live
            service that speaks itself says the failure line. ``attach_realtime`` gives it
            the prompt.
        transport: The call's transport. Its ``on_client_disconnected`` event notes that the
            caller hung up (``CALLER_HUNG_UP``), unless the call already has an end reason.
    """
    if call.bound:
        return

    if llm is not None:
        hide_async_guidance(llm)
    model = llm if llm is not None and speaks_itself(llm) else None
    if model is not None:
        follow_replies(model)

    async def say(line: str) -> None:
        """Queue a line. The hang-up queued after it lets it play first."""
        if model is not None:
            await speak(model, line, wait=True)
        else:
            await worker.queue_frame(TTSSpeakFrame(line))

    async def hang_up() -> None:
        """End the pipeline once what is queued has played."""
        await worker.queue_frame(EndFrame())

    call.bind(say=say, hang_up=hang_up, transcript=lambda: transcript(context))
    _follow_the_end(worker, call, transport)
    if isinstance(llm, ModelCallLLMService):
        llm.failure_is_final = True
    if llm is not None and not is_realtime(llm):
        _gate_requests(llm, call)
    elif llm is None and call.model_hooks:

        async def started(_worker: object, _frame: Frame) -> None:
            """Warn when nothing runs the call's model hooks, once the pipeline is built."""
            call.warn_skipped_model_hooks(SKIPPED_MODEL_HOOKS)

        worker.add_event_handler("on_pipeline_started", started)


def _follow_the_end(
    worker: PipelineWorker, call: CallHooks, transport: BaseTransport | None
) -> None:
    """Fail the call on a fatal error, end it with the pipeline, note the caller leaving."""

    async def failed(_worker: object, frame: ErrorFrame) -> None:
        """Fail the call on an error that leaves a service unable to work."""
        processor = frame.processor
        if frame.fatal or (processor is not None and not processor.is_usable):
            call.fail_soon(_component(processor), frame.exception or RuntimeError(frame.error))

    async def finished(_worker: object, frame: Frame) -> None:
        """End the call."""
        await call.end(ENDED)

    worker.add_event_handler("on_pipeline_error", failed)
    worker.add_event_handler("on_pipeline_finished", finished)
    if transport is not None:

        async def left(_transport: object, _client: object) -> None:
            """Note that the caller hung up."""
            call.caller_left()

        transport.add_event_handler("on_client_disconnected", left)


def _gate_requests(llm: LLMService[Any], call: CallHooks) -> None:
    """Hold each request for the tools on events, show their notes, run the model hooks."""
    requests = RequestHooks(llm, call, prompts=True)

    async def gate(_llm: object, frame: Frame) -> None:
        """Run the call's rules before one request."""
        if isinstance(frame, LLMContextFrame):
            await requests.give_back()
            await before_request(call, frame.context, main=True)
            await requests.before(frame)

    llm.add_event_handler("on_before_process_frame", gate)


def _component(processor: object) -> str:
    """Name the part of the call a processor is."""
    if isinstance(processor, STTService):
        return "stt"
    if isinstance(processor, TTSService):
        return "tts"
    if isinstance(processor, LLMService):
        return "llm"
    return "pipeline"


def transcript(context: LLMContext) -> list[dict[str, Any]]:
    """Return what the caller and the agent said, as ``{"role", "content"}`` messages."""
    return transcript_messages(context.get_messages(), drop_empty=True)


def _is_notes(message: object) -> bool:
    """Return True for the system message that holds the notes of tools on events."""
    return (
        isinstance(message, dict)
        and message.get("role") == "system"
        and str(message.get("content", "")).startswith(NOTES_HEADER)
    )


async def before_request(call: CallHooks, context: LLMContext, *, main: bool) -> None:
    """Wait for the tools on events, then put their notes at the end of a main request.

    Args:
        call: The call's hooks.
        context: The request's context.
        main: True for a main-agent request. A step's request gets no notes.
    """
    caller = newest_caller(context.get_messages(), skip_cue=True)
    newest = caller.get("content") if caller is not None else None
    await call.before_request(newest if isinstance(newest, str) else None)
    kept = [message for message in context.get_messages() if not _is_notes(message)]
    if main and (block := call.notes_block()) is not None:
        kept.append({"role": "system", "content": block})
    context.set_messages(cast(Any, kept))


async def open_call(
    worker: PipelineWorker,
    greeting: Greeting,
    call: CallHooks,
    *,
    context: LLMContext,
    llm: LLMService[Any] | None = None,
    hold: GreetingHold | None = None,
    transport: BaseTransport | None = None,
) -> bool:
    """Pass the checks before the greeting, then queue it. Call it once the client connects.

    Args:
        worker: The call's pipeline worker.
        greeting: How the call opens.
        call: The call's hooks, with the checks to pass.
        context: The shared context.
        llm: The LLM service of a pipeline without ``UnmuteFlows``. See ``attach_call``.
            Pass OpenAI's realtime or live service always, with ``UnmuteFlows`` too: it
            says the greeting and the failure line itself.
        hold: The greeting hold on the user aggregator. It lets go when the caller speaks
            first.
        transport: The call's transport. See ``attach_call``.

    Returns:
        True once the greeting is queued. False after a failed check: the failure line
        plays and the call ends.
    """
    attach_call(worker, call, context, llm=llm, transport=transport)
    if llm is None:
        call.warn_skipped_model_hooks(SKIPPED_MODEL_HOOKS)
    if hold is not None:
        hold.interruptible = greeting.interruptible
    if not await call.open():
        return False
    frames = greeting_frames(greeting, call.state, llm=llm)
    if greeting.speaks_first == "user" and llm is not None:
        await wait_for_caller(llm, context)
    elif llm is not None and is_realtime(llm) and greeting.line(call.state) is None:
        await call.before_request(None)
        await settled(llm)
    if (not frames or greeting.speaks_first == "user") and hold is not None:
        hold.release()
    await worker.queue_frames(frames)
    if hold is not None:
        hold.arm()
    return True
