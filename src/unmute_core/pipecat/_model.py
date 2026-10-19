"""A Pipecat LLM service that sends every request through a ``ModelCall``.

Native Pipecat LLM services need none of this: the task adapter fills their prompt. This
service is for a ``ModelCall`` such as ``unmute_core.slng.RouterModelCall`` or
``FallbackModelCall``. It reuses ``OpenAILLMService``'s stream handling, so text frames,
tool-call grouping and metrics stay Pipecat's own.

Pipecat records a tool call in the context with only its id, name and arguments. So the
service keeps each call's provider data, like a Gemini thought signature, by call id, and
puts it back as the call's ``extra_content`` on every request that replays the call. Only
the keys LiveKit sends back go out (``REPLAYED_PROVIDERS``), so both adapters replay the
same data.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import replace
from typing import Any, cast

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.http import TIMEOUT_EXCEPTIONS
from pipecat.utils.types import assert_given

from unmute_core.call import CallHooks
from unmute_core.model import (
    EXTRA_CONTENT,
    ModelCall,
    ModelEvent,
    ModelRequest,
    ReasoningFilter,
    TextDelta,
    ToolCall,
    Usage,
    close_model,
    model_name,
    replayed_extra,
    template_variables,
    tool_choice,
)
from unmute_core.pipecat._services import llm_settings
from unmute_core.state import CallState

logger = logging.getLogger(__name__)


class ModelCallLLMService(OpenAILLMService):
    """Runs each Pipecat model request through a ``ModelCall``.

    The prompt stays unfilled here. The request carries the variables, rendered from the
    call state for the current prompt site, and the ``ModelCall`` decides who fills them.

    Attributes:
        model_call: Where the requests go.
        state: The call state, or None for an agent without one.
        site: The prompt site of the next request: ``""`` for the main agent, or
            ``task:<name>``. The task adapter sets it.
        failure_is_final: True makes a failed model call leave the service unusable, so the
            call fails. ``attach_call(llm=...)`` and ``UnmuteFlows(call=...)`` set it. False
            reports the failure as Pipecat does, as one the service recovers from. A call
            that fails after its reply began is never final: the caller already heard the
            start, so the reply ends there and the call goes on.
        call: The call's hooks. Each request runs its ``before_model`` and ``wrap_model``
            hooks. ``attach_call(llm=...)`` and ``UnmuteFlows(call=...)`` set it.
        last_request: The newest request as it went out, after the hooks.
        trace_headers: Returns the trace headers to send with the next request, or None to
            send none. ``TelemetryObserver`` sets it, so a router's spans join the turn's
            trace. They win over a request's own headers.
        tool_extras: The provider data of each tool call the model call streamed, by
            call id. Each request puts it back on the calls it replays.
        switched: Model calls ``hand_off(model=...)`` switched in. ``cleanup`` closes each.
        owned: Other model calls the service closes with its own, like the summarizer given
            to ``UnmuteFlows``.
    """

    def __init__(
        self,
        model_call: ModelCall,
        *,
        state: CallState | None = None,
        call: CallHooks | None = None,
        prompts_as_written: bool = False,
        **kwargs: Any,
    ) -> None:
        """Wrap the model call.

        Args:
            model_call: Where the requests go.
            state: The call state, read for each request's variables.
            call: The call's hooks, run on each request.
            prompts_as_written: True sends each prompt as written, for prompts that are not
                templates: a ``{{name}}`` in them is plain text, not a placeholder.
            **kwargs: ``OpenAILLMService`` arguments, such as ``settings``. A key or a
                base URL is not used.
        """
        self.model_call = model_call
        self.state = state
        self.prompts_as_written = prompts_as_written
        self.site = ""
        self.failure_is_final = False
        self.call = call
        self.last_request: ModelRequest | None = None
        self.trace_headers: Callable[[], Mapping[str, str]] | None = None
        self.tool_extras = ToolExtras()
        self.switched: list[ModelCall] = []
        self.owned: list[ModelCall] = []
        self._first_model_call = model_call
        self._replies: dict[asyncio.Task[Any] | None, list[bool]] = {}
        super().__init__(**kwargs)

    async def cleanup(self) -> None:
        """Clean up the service, then close each model call it used, once.

        That is the one it was built with, the current one, each one switched in and each
        one in ``owned``.
        """
        await super().cleanup()
        closing = {
            id(model): model
            for model in (self._first_model_call, self.model_call, *self.switched, *self.owned)
        }
        for model in closing.values():
            await close_model(model)

    def create_client(self, *args: Any, **kwargs: Any) -> None:
        """Build no OpenAI client. The model call owns the connection."""
        return None

    async def get_chat_completions(self, context: LLMContext) -> Any:
        """Stream one request through the model call, as OpenAI chunks.

        Each replayed tool call gets back the provider data it streamed with, before the
        ``before_model`` hooks run.
        """
        params = self.get_llm_adapter().get_llm_invocation_params(
            context,
            system_instruction=assert_given(llm_settings(self).system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )
        messages = with_tool_extras(
            [dict(message) for message in params["messages"]], self.tool_extras
        )
        tools = params.get("tools")
        request = ModelRequest(
            messages=messages,
            tools=[dict(tool) for tool in tools] if isinstance(tools, list) else [],
            site=self.site,
            tool_choice=tool_choice(params.get("tool_choice")),
            model=model_name(self.model_call),
        )
        hooks = self.call.hooks if self.call is not None else None
        if hooks is not None:
            request = await hooks.run_before_model(request)
        request = replace(
            request,
            variables=template_variables(
                request.messages, self.state, self.site, as_written=self.prompts_as_written
            ),
        )
        if self.trace_headers is not None:
            request = replace(request, headers=request.headers | dict(self.trace_headers()))
        self.last_request = request
        events = (
            hooks.run_model(request, self.model_call.stream)
            if hooks is not None
            else self.model_call.stream(request)
        )
        started = self._replies.get(asyncio.current_task())
        return _chunks(
            events,
            self.get_full_model_name() or "",
            self.tool_extras,
            (lambda: started.append(True)) if started is not None else (lambda: None),
        )

    async def _process_context(self, context: LLMContext) -> None:
        """Run one request. With ``failure_is_final``, a failure leaves the service unusable.

        A model call that raises has already used its own retries and fallbacks, so the
        call cannot go on. A timeout still fires ``on_completion_timeout`` first. A
        failure after that request's reply began is the exception: the error is reported as one the
        service recovers from, the reply ends as it stands, and the call goes on.
        """
        if not self.failure_is_final:
            await super()._process_context(context)
            return
        started: list[bool] = []
        task = asyncio.current_task()
        self._replies[task] = started
        try:
            await super()._process_context(context)
        except Exception as error:  # noqa: BLE001 - reported upstream as an ErrorFrame
            if isinstance(error, TIMEOUT_EXCEPTIONS):
                await self._call_event_handler("on_completion_timeout")
            if started:
                logger.error("the model call failed after its reply began: %s", error)
                await self.push_error(
                    error_msg=f"The model call failed after its reply began: {error}",
                    exception=error,
                )
                return
            await self.push_error(
                error_msg=f"The model call failed: {error}",
                exception=error,
                force_treat_as_permanent=True,
            )
        finally:
            self._replies.pop(task, None)


def _chunk(delta: ChoiceDelta, model: str) -> ChatCompletionChunk:
    """Wrap one delta as the chunk ``OpenAILLMService`` reads."""
    return ChatCompletionChunk(
        id="",
        choices=[Choice(index=0, delta=delta)],
        created=0,
        model=model,
        object="chat.completion.chunk",
    )


class ToolExtras(dict[str, Mapping[str, Any]]):
    """Call id to the provider data a tool call streamed with, to send back when it is replayed."""

    def restore(self, messages: list[Any]) -> list[Any]:
        """Return the messages with each assistant tool call's provider data put back.

        Only the keys of ``REPLAYED_PROVIDERS`` go back, as LiveKit sends them. A call that
        already has an ``extra_content`` keeps it.

        Args:
            messages: The chat in OpenAI format. A provider's own message, not a dict,
                stays as it is.

        Returns:
            A new list. Messages with nothing to put back are the same objects.
        """
        return [
            {**message, "tool_calls": calls}
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and (calls := self._restored(message.get("tool_calls"))) is not None
            else message
            for message in messages
        ]

    def _restored(self, calls: object) -> list[Any] | None:
        """Return the tool calls with their provider data put back, or None when none changes."""
        if not isinstance(calls, list) or not any(self._missing(call) for call in calls):
            return None
        return [
            {**call, EXTRA_CONTENT: replayed_extra(self[call["id"]])}
            if self._missing(call)
            else call
            for call in calls
        ]

    def _missing(self, call: object) -> bool:
        """Return whether a tool call lacks provider data this holds for it to send back."""
        return (
            isinstance(call, dict)
            and EXTRA_CONTENT not in call
            and bool(replayed_extra(self.get(call.get("id", ""), {})))
        )

    def keep(self, call: ToolCall) -> None:
        """Keep a tool call's provider data by its call id, to send back when it is replayed.

        A call id seen again with none drops the older entry, so an old signature never
        rides on a new call.
        """
        if call.extra:
            self[call.id] = call.extra
        else:
            self.pop(call.id, None)


def with_tool_extras(messages: list[Any], extras: Mapping[str, Mapping[str, Any]]) -> list[Any]:
    """Return the messages with each assistant tool call's provider data put back.

    See ``ToolExtras.restore``.
    """
    return ToolExtras(extras).restore(messages)


def keep_extra(extras: dict[str, Mapping[str, Any]], call: ToolCall) -> None:
    """Keep a tool call's provider data by its call id. See ``ToolExtras.keep``."""
    ToolExtras.keep(cast(ToolExtras, extras), call)


async def _chunks(
    events: AsyncIterator[ModelEvent],
    model: str,
    extras: ToolExtras,
    started: Callable[[], None] = lambda: None,
) -> AsyncIterator[Any]:
    """Yield the model call's events as OpenAI chunks, reasoning tags taken out of the text.

    Each tool call's provider data goes into ``extras``, through ``ToolExtras.keep``.
    ``started`` is called before the first chunk of text or of a tool call goes out.
    """
    index = 0
    speech = ReasoningFilter()
    async for event in events:
        if isinstance(event, TextDelta):
            if text := speech.feed(event.text):
                started()
                yield _chunk(ChoiceDelta(content=text), model)
            continue
        if isinstance(event, Usage):
            usage = CompletionUsage(
                completion_tokens=event.output_tokens,
                prompt_tokens=event.input_tokens,
                total_tokens=event.input_tokens + event.output_tokens,
                prompt_tokens_details=(
                    PromptTokensDetails(cached_tokens=event.cached_input_tokens)
                    if event.cached_input_tokens is not None
                    else None
                ),
            )
            yield ChatCompletionChunk(
                id=event.response_id or "",
                choices=[],
                created=0,
                model=model,
                object="chat.completion.chunk",
                usage=usage,
            )
            continue
        extras.keep(event)
        started()
        call = ChoiceDeltaToolCall(
            index=index,
            id=event.id,
            type="function",
            function=ChoiceDeltaToolCallFunction(name=event.name, arguments=event.arguments),
        )
        index += 1
        yield _chunk(ChoiceDelta(tool_calls=[call]), model)
    if tail := speech.flush():
        started()
        yield _chunk(ChoiceDelta(content=tail), model)
