"""The call's model hooks on the requests a Pipecat LLM service makes in the pipeline.

Pipecat has no hook around one request of a service, and the library never wraps a
service. So each request's changes are lent and given back:

- changed messages go out on a copy of the context, so the shared chat keeps none;
- a changed system prompt and ``params`` are set on the service for this request, then
  set back when it ends, or at the next request when an interruption cut it off.

A ``ModelCallLLMService`` runs ``before_model`` and ``wrap_model`` itself, on the request
it builds. Every service gets ``after_model`` from the frames it pushes. Its tool calls
carry the provider data the service kept for them, as on LiveKit.
"""

import asyncio
import json
import logging
import weakref
from collections.abc import Callable
from typing import Any, cast

from pipecat.frames.frames import (
    Frame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pydantic import SkipValidation
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.call import CallHooks
from unmute_core.hooks import ModelResponse, PromptInfo
from unmute_core.model import ModelRequest, ToolCall, model_name
from unmute_core.pipecat._model import ModelCallLLMService
from unmute_core.pipecat._services import llm_settings

logger = logging.getLogger(__name__)

_SERVED: "weakref.WeakSet[LLMService[Any]]" = weakref.WeakSet()
"""The LLM services that run a call's hooks. Each may serve only one call."""


def service_model(llm: object) -> str:
    """Return the model a Pipecat LLM service sends, like ``gpt-4.1-mini``, or ``unknown``."""
    if isinstance(llm, ModelCallLLMService):
        return model_name(llm.model_call)
    settings = getattr(llm, "_settings", None)
    return str(getattr(settings, "model", "") or "unknown")


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class _LentPrompt:
    """The system prompt a request borrowed: what it was, and what the request set."""

    original: str | None
    prompt: str


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class _Lent:
    """The ``params`` a request borrowed: the settings' own ``extra`` and the one lent."""

    original: SkipValidation[dict[str, Any]]
    lent: SkipValidation[dict[str, Any]]


class RequestHooks:
    """Runs the call's model hooks on one LLM service's pipeline requests.

    The owner, ``UnmuteFlows`` or ``attach_call``, calls ``give_back`` first and ``before``
    last in its own handler of each ``LLMContextFrame``, so the hooks see the request as it
    goes out.
    """

    def __init__(
        self,
        llm: LLMService[Any],
        call: CallHooks,
        *,
        site: Callable[[], str] = lambda: "",
        prompts: bool = False,
    ) -> None:
        """Watch the service's frames.

        Args:
            llm: The LLM service.
            call: The call's hooks.
            site: Returns the prompt site of the next request.
            prompts: Run the ``prompt`` hooks on the service's own system prompt. Off when
                the owner builds the prompts itself, as ``UnmuteFlows`` does.

        Raises:
            ValueError: If the service already runs hooks: each call needs its own service,
                and ``attach_call(llm=)`` is only for a pipeline without ``UnmuteFlows``.
        """
        if llm in _SERVED:
            raise ValueError(
                "this LLM service already runs a call's hooks: build one LLM service per call, "
                "and pass llm= to attach_call only in a pipeline without UnmuteFlows"
            )
        _SERVED.add(llm)
        call.note_model_requests()
        self._llm = llm
        self._call = call
        self._site = site
        self._prompts = prompts
        self._routed = isinstance(llm, ModelCallLLMService)
        self._system: _LentPrompt | None = None
        self._extra: _Lent | None = None
        self._request: ModelRequest | None = None
        self._text: list[str] = []
        self._calls: list[ToolCall] = []
        self._replying = False
        self._warned = False
        self._after: set[asyncio.Task[None]] = set()
        if isinstance(llm, ModelCallLLMService):
            llm.call = call
        llm.add_event_handler("on_after_process_frame", self._processed)
        llm.add_event_handler("on_before_push_frame", self._pushed)

    async def give_back(self) -> None:
        """Set back the system prompt and ``params`` a request borrowed, if still borrowed."""
        if (lent := self._extra) is not None:
            self._extra = None
            settings = self._llm.settings
            if settings.extra is lent.lent:
                settings.extra = lent.original
        if (system := self._system) is not None:
            self._system = None
            if getattr(self._llm, "_base_system_instruction", None) == system.prompt:
                await self._llm._update_settings(LLMSettings(system_instruction=system.original))

    async def before(self, frame: LLMContextFrame) -> None:
        """Run the ``prompt`` and ``before_model`` hooks on the request about to go out.

        A hook that raises fails the call, as a failed model call does.
        """
        try:
            await self._before(frame)
        except Exception as error:
            self._call.fail_soon("llm", error)
            raise

    async def _before(self, frame: LLMContextFrame) -> None:
        """Run the hooks. See ``before``."""
        hooks = self._call.hooks
        prompted = self._prompts and hooks.has("prompt")
        if not (prompted or hooks.has("before_model") or hooks.has("after_model")):
            return
        site = self._site()
        system = self._base()
        name = service_model(self._llm)
        if prompted and system is not None:
            await self._lend_system(
                system, hooks.run_prompt(system, PromptInfo("agent", site, name))
            )
            system = self._base()
        messages = [
            dict(message) for message in frame.context.get_messages() if isinstance(message, dict)
        ]
        head = [{"role": "system", "content": system}] if system is not None else []
        request = ModelRequest(messages=[*head, *messages], site=site, model=name)
        self._request = request
        self._text, self._calls, self._replying = [], [], True
        if self._routed or not hooks.has("before_model"):
            return
        hooked = await hooks.run_before_model(request)
        self._request = hooked
        if hooked.tools or hooked.tool_choice is not None:
            self._warn("a before_model hook set tools: a native Pipecat LLM keeps its own")
        if hooked.messages is not request.messages:
            await self._replace_messages(frame, list(hooked.messages), head, system)
        if hooked.params:
            extra = self._llm.settings.extra
            body = dict(extra.get("extra_body") or {}) | hooked.params
            lent = extra | {"extra_body": body}
            self._llm.settings.extra = lent
            self._extra = _Lent(extra, lent)

    async def _replace_messages(
        self,
        frame: LLMContextFrame,
        messages: list[dict[str, Any]],
        head: list[dict[str, Any]],
        system: str | None,
    ) -> None:
        """Send a hook's messages on a copy of the context. A changed system prompt is lent."""
        context = frame.context
        if len(messages) != len(context.get_messages()) + len(head) and any(
            not isinstance(message, dict) for message in context.get_messages()
        ):
            self._warn("a before_model hook changed a chat with provider messages: kept as it was")
            return
        if head:
            if not messages or messages[0].get("role") != "system":
                self._warn("a before_model hook moved the system prompt: Pipecat needs it first")
                return
            prompt = messages.pop(0).get("content")
            if isinstance(prompt, str):
                await self._lend_system(system, prompt)
        frame.context = LLMContext(
            messages=cast(Any, messages), tools=context.tools, tool_choice=context.tool_choice
        )

    def _base(self) -> str | None:
        """Return the service's own system prompt, before Pipecat adds its guidance to it.

        Pipecat composes the prompt it sends from this base and its own parts, like the
        async-tool guidance, on every request. Hooks change the base, so nothing doubles.
        """
        base = getattr(self._llm, "_base_system_instruction", None)
        if base is None:
            base = llm_settings(self._llm).system_instruction
        return base if isinstance(base, str) else None

    async def _lend_system(self, original: str | None, prompt: str) -> None:
        """Set the system prompt for this request, to give back once it ends."""
        if prompt == original:
            return
        await self._llm._update_settings(LLMSettings(system_instruction=prompt))
        if self._system is None:
            self._system = _LentPrompt(original, prompt)
        else:
            self._system = _LentPrompt(self._system.original, prompt)

    def _warn(self, message: str) -> None:
        """Log a limit of native Pipecat once a call."""
        if not self._warned:
            self._warned = True
            logger.warning(message)

    async def _processed(self, _llm: object, frame: Frame) -> None:
        """Give back what a request borrowed once it ran or was cut off, and note a cut."""
        if isinstance(frame, LLMContextFrame):
            await self.give_back()
        elif isinstance(frame, InterruptionFrame):
            await self.give_back()
            if self._replying:
                self._answered(interrupted=True)

    async def _pushed(self, _llm: object, frame: Frame) -> None:
        """Follow the reply's text and tool calls, and tell ``after_model`` when it ends."""
        if isinstance(frame, LLMFullResponseStartFrame):
            self._text, self._calls = [], []
        elif isinstance(frame, LLMTextFrame):
            self._text.append(frame.text)
        elif isinstance(frame, FunctionCallsStartedFrame):
            extras = self._llm.tool_extras if isinstance(self._llm, ModelCallLLMService) else {}
            self._calls.extend(
                ToolCall(
                    call.tool_call_id,
                    call.function_name,
                    json.dumps(call.arguments, ensure_ascii=False),
                    extras.get(call.tool_call_id) or {},
                )
                for call in frame.function_calls
            )
        elif isinstance(frame, LLMFullResponseEndFrame) and self._replying:
            self._answered(interrupted=False)

    def _answered(self, *, interrupted: bool) -> None:
        """Tell the ``after_model`` hooks how the request ended, in a task of their own."""
        self._replying = False
        hooks = self._call.hooks
        request = self._llm.last_request if isinstance(self._llm, ModelCallLLMService) else None
        request = request or self._request
        if request is None or not hooks.has("after_model"):
            return
        response = ModelResponse("".join(self._text), tuple(self._calls), interrupted=interrupted)
        task = asyncio.ensure_future(hooks.run_after_model(request, response))
        self._after.add(task)
        task.add_done_callback(self._after.discard)
