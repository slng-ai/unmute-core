"""OpenAI's realtime and live services in a Pipecat pipeline: who speaks a line, and how.

A realtime service keeps the conversation on its own server. It reads the local context
once, on the first ``LLMContextFrame``. After that, ``LLMRunFrame`` and
``LLMMessagesAppendFrame`` start no reply (services/openai/realtime/llm.py:677-698). So a
line said mid-call goes to the server as a client event, as Unmute's realtime idle nudge
does (unmute@2c78a5a templates/pipecat_v1/call.py.tmpl:263-290).
"""

import asyncio
import contextlib
import weakref
from collections.abc import Mapping
from typing import Any, ClassVar, TypeGuard, cast

from pipecat.frames.frames import (
    Frame,
    FunctionCallsStartedFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMSetToolsFrame,
    LLMUpdateSettingsFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.live import events as live_events
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.realtime import events as realtime_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.settings import LLMSettings

from unmute_core.call import CallHooks
from unmute_core.pipecat._services import hide_async_guidance, llm_settings
from unmute_core.realtime import (
    SETTLE_WAIT_SECONDS,
    check_realtime,
    realtime_instructions,
    say_line,
)
from unmute_core.state import CallState
from unmute_core.tools import values_block

_RESENDS: weakref.WeakKeyDictionary[LLMService[Any], set[asyncio.Task[None]]] = (
    weakref.WeakKeyDictionary()
)
"""The tasks that send a service its instructions again, per service."""

REPLY_WAIT_SECONDS = 10.0
"""The longest a line waits for the running reply to end, and then for its own."""


class _Replies:
    """Tells when the service's replies start and end. One per service.

    The server refuses a new reply while one runs. A tool runs inside the reply that
    called it, so its announce line waits for that reply to end. A reply that only calls
    a tool pushes no start frame, only its end (services/openai/realtime/llm.py:955,
    1035), so its tool call marks it as running.
    """

    def __init__(self, service: LLMService[Any]) -> None:
        """Watch the frames the service pushes."""
        self.idle = asyncio.Event()
        self.idle.set()
        self.started = 0
        self.changed = asyncio.Event()
        self._calls: set[int] = set()
        service.add_event_handler("on_after_push_frame", self._pushed)

    async def _pushed(self, _service: object, frame: Frame) -> None:
        """Note a reply starting or ending.

        A tool call is pushed both ways. Only its first copy counts: the second could
        come after the reply's end.
        """
        if isinstance(frame, FunctionCallsStartedFrame):
            if frame.broadcast_sibling_id in self._calls:
                return
            self._calls.add(frame.id)
        if isinstance(frame, LLMFullResponseStartFrame | FunctionCallsStartedFrame):
            if self.idle.is_set():
                self.started += 1
            self.idle.clear()
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.idle.set()
        self.changed.set()

    async def reply_after(self, count: int) -> None:
        """Wait until a reply that started after the first ``count`` replies has ended."""
        while self.started <= count or not self.idle.is_set():
            self.changed.clear()
            await self.changed.wait()


class RealtimeVoice:
    """One OpenAI realtime or live service, and how a line reaches its model.

    ``RealtimeVoice(service)`` is cheap: the replies each service follows sit in one table
    that all of a service's ``RealtimeVoice`` objects share.
    """

    _replies: ClassVar[weakref.WeakKeyDictionary[LLMService[Any], _Replies]] = (
        weakref.WeakKeyDictionary()
    )

    def __init__(self, service: LLMService[Any]) -> None:
        """Wrap the service."""
        self.service = service

    @staticmethod
    def is_realtime(service: object) -> bool:
        """Return True for OpenAI's realtime or live service, or a subclass, like Azure's."""
        return isinstance(service, OpenAIRealtimeLLMService | OpenAILiveLLMService)

    @property
    def fixed_instructions(self) -> bool:
        """True for a service that fixes its instructions when the session starts."""
        return isinstance(self.service, OpenAILiveLLMService)

    @property
    def speaks_itself(self) -> bool:
        """True when the service speaks, so no TTS can say a fixed line.

        A realtime service set to answer in text only has a TTS after it: a half cascade.
        """
        service = self.service
        if isinstance(service, OpenAILiveLLMService):
            return True
        if not isinstance(service, OpenAIRealtimeLLMService):
            return False
        properties = getattr(service.settings, "session_properties", None)
        modalities = getattr(properties, "output_modalities", None)
        return modalities is None or "audio" in modalities

    def follow_replies(self) -> _Replies:
        """Start following the service's replies, once, and return what follows them."""
        if self.service not in self._replies:
            self._replies[self.service] = _Replies(self.service)
        return self._replies[self.service]

    async def wait_for_caller(self, context: LLMContext) -> None:
        """Ready a realtime service for a call the caller opens, with no reply of its own.

        The service sets its conversation up on the first ``LLMContextFrame``, and asks for
        a reply then (services/openai/realtime/llm.py:676-684). When the caller speaks
        first, that frame comes in the middle of their turn. The service would send the
        turn again as a "previously saved conversation" and ask for a second reply, and the
        tools would reach the model only then. So it gets the context at once, then the
        tools, which the service registers and sends. A live service is left alone: its
        session opens on the first frame by design.
        """
        service = self.service
        if not isinstance(service, OpenAIRealtimeLLMService) or service._context is not None:
            return
        service._context = context
        service._llm_needs_conversation_setup = bool(context.get_messages())
        await service.queue_frame(LLMSetToolsFrame(tools=context.tools))

    async def ask(self, instruction: str | None, *, wait: bool = False) -> None:
        """Have the model reply now, told ``instruction`` on top of its own instructions.

        A realtime reply carries it for that reply only, as LiveKit's ``generate_reply``
        does. It waits for a running reply to end first. A live model reads it as
        commentary, which it speaks in its own words.

        Args:
            instruction: What the model is told. None replies on its own instructions: a
                live model has nothing to read, so it is skipped.
            wait: Return once the reply has ended, at most ``REPLY_WAIT_SECONDS`` later.

        Raises:
            TypeError: If the service is neither.
        """
        service = self.service
        if not self.is_realtime(service):
            raise TypeError(f"{type(service).__name__} is not OpenAI's realtime or live service")
        replies = self.follow_replies()
        if isinstance(service, OpenAILiveLLMService):
            if instruction is None:
                return
            event: object = live_events.SessionCommentaryAppendEvent(
                delegation_id=None, content=instruction
            )
        else:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(replies.idle.wait(), REPLY_WAIT_SECONDS)
            system = llm_settings(service).system_instruction
            parts = [part for part in (system, instruction) if isinstance(part, str) and part]
            event = realtime_events.ResponseCreateEvent(
                response=realtime_events.ResponseProperties(instructions="\n".join(parts) or None)
            )
        started = replies.started
        await cast(Any, service).send_client_event(event)
        if wait:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(replies.reply_after(started), REPLY_WAIT_SECONDS)

    async def speak(self, line: str, *, wait: bool = False) -> None:
        """Have the model say ``line``, word for word as far as it can. See ``ask``."""
        live = isinstance(self.service, OpenAILiveLLMService)
        await self.ask(line if live else say_line(line), wait=wait)


def is_realtime(service: object) -> bool:
    """Return True for OpenAI's realtime or live service, or a subclass, like Azure's."""
    return RealtimeVoice.is_realtime(service)


def fixed_instructions(service: object) -> bool:
    """Return True for a service that fixes its instructions when the session starts."""
    return isinstance(service, OpenAILiveLLMService)


def speaks_itself(service: object) -> bool:
    """Return True when the service speaks, so no TTS can say a fixed line.

    A realtime service set to answer in text only has a TTS after it: a half cascade.
    """
    return is_realtime(service) and RealtimeVoice(cast(LLMService[Any], service)).speaks_itself


async def wait_for_caller(service: LLMService[Any], context: LLMContext) -> None:
    """Ready a realtime service for a call the caller opens. See ``RealtimeVoice``."""
    await RealtimeVoice(service).wait_for_caller(context)


def withhold_result(service: object, call_id: str) -> None:
    """Keep a realtime service from sending a tool's result, and so from asking for a reply.

    The service reads each result in the context and sends it, and a result starts another
    reply. A call that ends the pipeline needs no reply, but its result still goes through
    Pipecat, so the tool span, the transcript and the dev record show the tool. A service
    that is not realtime is left alone: ``run_llm=False`` is enough for it.

    Args:
        service: The call's LLM service.
        call_id: The tool call whose result stays local.
    """
    if isinstance(service, OpenAIRealtimeLLMService):
        service._completed_tool_calls.add(call_id)
    elif isinstance(service, OpenAILiveLLMService):
        service._open_function_calls.pop(call_id, None)


def can_hold(service: object) -> TypeGuard[OpenAIRealtimeLLMService]:
    """Return True when the service sends a tool's result only on its next context.

    OpenAI's realtime service does, so a result can wait there. The live service sends
    each result at once.
    """
    return isinstance(service, OpenAIRealtimeLLMService)


def hold_result(service: object, call_id: str, *, held: bool) -> None:
    """Hold back a tool's result from a realtime service, or hand it to the service again.

    A result the service has not sent yet can wait: the service sends it, and asks for a
    reply, on the next context it reads after the result is back in the context. Handing
    it back makes it eligible again. Only OpenAI's realtime service holds a result this
    way. The live service answers at once and has nothing to hold.

    Args:
        service: The call's LLM service.
        call_id: The tool call.
        held: True to hold the result back, False to release it.
    """
    if can_hold(service) and held:
        service._completed_tool_calls.add(call_id)
    elif can_hold(service):
        service._completed_tool_calls.discard(call_id)


def keep_resend(service: LLMService[Any], task: asyncio.Task[None]) -> None:
    """Note a task that sends a realtime service its instructions, until it is done."""
    pending = _RESENDS.setdefault(service, set())
    pending.add(task)
    task.add_done_callback(pending.discard)


async def settled(service: LLMService[Any]) -> None:
    """Wait until every instruction update queued for a realtime service is on its way.

    The update frames go straight to the service, so they are ahead of any frame queued
    on the pipeline after this returns. It waits at most ``SETTLE_WAIT_SECONDS``.
    """
    pending = _RESENDS.get(service, ())
    if pending:
        await asyncio.wait(pending, timeout=SETTLE_WAIT_SECONDS)


def follow_replies(service: LLMService[Any]) -> None:
    """Start following the service's replies. ``attach_realtime`` does it."""
    RealtimeVoice(service).follow_replies()


async def reply_ended(service: LLMService[Any]) -> None:
    """Wait until the service's running reply has ended, at most ``REPLY_WAIT_SECONDS``.

    A realtime service runs each tool call as soon as its arguments are done, before the
    reply has named its other calls. A result sent then asks for another reply, even when
    a later call of the same reply ends the call.
    """
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(
            RealtimeVoice(service).follow_replies().idle.wait(), REPLY_WAIT_SECONDS
        )


async def ask(service: LLMService[Any], instruction: str | None, *, wait: bool = False) -> None:
    """Have the model reply now, told ``instruction`` on top of its own instructions.

    See ``RealtimeVoice.ask``.

    Raises:
        TypeError: If the service is neither OpenAI's realtime nor live service.
    """
    await RealtimeVoice(service).ask(instruction, wait=wait)


async def speak(service: LLMService[Any], line: str, *, wait: bool = False) -> None:
    """Have the model say ``line``, word for word as far as it can. See ``ask``."""
    await RealtimeVoice(service).speak(line, wait=wait)


async def attach_realtime(
    service: LLMService[Any],
    worker: PipelineWorker,
    prompt: str,
    state: CallState | None = None,
    *,
    call: CallHooks | None = None,
    save_values: bool = False,
    owners: Mapping[str, str] | None = None,
) -> None:
    """Run OpenAI's realtime or live service as the agent: its instructions and its lines.

    Await it before the pipeline runs, for a pipeline without ``UnmuteFlows``. It refuses
    what the model cannot do, then queues the filled instructions on the service: they
    reach the model before anything queued later. Each caller's words run the call's tools
    on events, and their notes follow. The model answers meanwhile: nothing waits for them. A realtime model hears
    them again after each save. A live model reads them once, as it starts.

    Args:
        service: The call's realtime or live service.
        worker: The call's worker. Its end stops following the state.
        prompt: The agent's prompt, placeholders still in it.
        state: The call state the placeholders read.
        call: The call's hooks: its ``prompt`` hooks and the notes of its tools on events.
        save_values: The agent shows its values after the prompt. A State with no values
            shows none.
        owners: Value name to the task that saves it, for the values block.

    Raises:
        TypeError: If the service is not OpenAI's realtime or live service.
        ValueError: If the call has model hooks, or ``save_values`` on a live model.
    """
    if not is_realtime(service):
        raise TypeError(f"{type(service).__name__} is not OpenAI's realtime or live service")
    fixed = fixed_instructions(service)
    check_realtime(
        fixed_instructions=fixed,
        save_values=save_values,
        hooks=call.hooks if call is not None else None,
    )
    follow_replies(service)
    hide_async_guidance(service)

    def text() -> str:
        """Return the instructions now."""
        model = service.settings.model
        return realtime_instructions(
            prompt,
            state,
            hooks=call.hooks if call is not None else None,
            model=model if isinstance(model, str) else "unknown",
            blocks=(
                call.notes_block() if call is not None else None,
                values_block(state, owners=owners)
                if save_values and state is not None and state.names
                else None,
            ),
        )

    def instruct() -> Frame:
        """Return the frame that sends the instructions now."""
        return LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=text()))

    await service.queue_frame(instruct())

    async def resend(caller: str | None) -> None:
        """Run the tools on events for a caller's words, then send the instructions."""
        if call is not None and caller is not None:
            await call.before_request(caller)
        if not fixed:
            await service.queue_frame(instruct())

    def soon(caller: str | None) -> None:
        """Start ``resend`` beside the model's reply."""
        keep_resend(service, asyncio.create_task(resend(caller)))

    async def heard(_service: object, frame: Frame) -> None:
        """Pass the caller's words on."""
        if isinstance(frame, TranscriptionFrame):
            soon(frame.text)

    if call is not None:
        service.add_event_handler("on_after_push_frame", heard)
    if fixed or state is None:
        return
    unlisten = state.listen(lambda _event: soon(None))

    async def finished(_worker: object, _frame: object) -> None:
        """Stop following the state."""
        unlisten()

    worker.add_event_handler("on_pipeline_finished", finished)
