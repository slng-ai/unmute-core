"""Every model request of an Unmute agent and its steps, from LiveKit's ``llm_node``.

Each request carries the prompt of its agent or step as it reads now, and the values of the
call state that prompt names. With the session's own LLM plugin, the prompt is filled here
and the plugin's chunks pass through unchanged, so provider data like a Gemini thought
signature survives. With a ``ModelCall``, the request goes out in OpenAI format with the
values beside it, and the call's events come back as chunks. A ``ToolCall``'s ``extra``
becomes the chunk's ``FunctionToolCall.extra``: LiveKit keeps it on the history's
``FunctionCall`` and sends it back as the call's ``extra_content`` (llm/_provider_format/
openai.py:51), so the signature survives this way too.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from typing import Any, cast

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    NotGivenOr,
    llm,
    utils,
)
from livekit.agents.metrics import LLMMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN
from livekit.agents.voice import Agent, AgentSession, ModelSettings
from opentelemetry.trace import Span

from unmute_core.call import CallHooks
from unmute_core.handoff import (
    SUMMARY_PROMPT,
    SUMMARY_SITE,
    SUMMARY_TIMEOUT_SECONDS,
    summary_request,
)
from unmute_core.hooks import Hooks, ModelResponse, PromptInfo, PromptKind
from unmute_core.livekit._dev import DevObserver, dev_observer, side_request
from unmute_core.livekit._history import CallerHistory
from unmute_core.livekit._trace import SessionTrace
from unmute_core.model import (
    EXTRA_CONTENT,
    ModelAnswer,
    ModelCall,
    ModelEvent,
    ModelRequest,
    NativeModelCall,
    ReasoningFilter,
    TextDelta,
    ToolCall,
    Usage,
    answered_model,
    close_model,
    fill_messages,
    http_status,
    model_name,
    provider_name,
    retries_itself,
    template_variables,
    tool_choice,
    variable_key_max,
)
from unmute_core.model._failover import transport_failure
from unmute_core.state import CallState
from unmute_core.telemetry import Telemetry
from unmute_core.tools import is_values, values_block

logger = logging.getLogger(__name__)

RUNNING_PLACEHOLDER_KEY = "__lk_running_placeholder__"
"""The flag on a running tool call LiveKit adds to a request (voice/generation.py:70)."""
INSTRUCTIONS_MESSAGE_ID = "lk.agent_task.instructions"
"""The id of the message that holds an agent's prompt (voice/generation.py:1280)."""
PROMPT_ROLES = frozenset({"system", "developer"})


class ModelCallLLM(llm.LLM):
    """A ``ModelCall`` as a LiveKit LLM, like the SLNG Context Router.

    Give it to any agent or session: ``AgentSession(llm=ModelCallLLM(router, state=state))``.
    Each request fills nothing itself: it goes out with the values of the call state its
    prompt names, and the model call fills the prompt. ``UnmuteAgent`` uses one too, and
    sends its requests through its own ``llm_node``.

    Attributes:
        site: The prompt site of the next request: ``""`` for the main prompt, or
            ``task:<name>``.
    """

    def __init__(
        self,
        model: ModelCall,
        *,
        state: CallState | None = None,
        trace: SessionTrace | None = None,
        site: str = "",
        call: CallHooks | None = None,
        prompts_as_written: bool = False,
    ) -> None:
        """Keep the model call and what each request reads.

        Args:
            model: The model call that answers.
            state: The call state the prompt's placeholders read, or None.
            trace: The session's trace from ``trace_session``. Each request is then an
                ``llm`` span in its turn.
            site: The prompt site of each request, until changed.
            call: The call's hooks. Each request waits for the tools on events and ends
                with their notes. None uses the trace's.
            prompts_as_written: True sends each prompt as written, for prompts that are not
                templates: a ``{{name}}`` in them is plain text, not a placeholder.
        """
        super().__init__()
        self._model = model
        self._node = ModelNode(
            state=state,
            model=model,
            trace=trace,
            call=call or (trace.call if trace else None),
            prompts_as_written=prompts_as_written,
        )
        self.site = site

    @property
    def model(self) -> str:
        """The model name the call sends, or ``unknown``."""
        return model_name(self._model)

    @property
    def variable_key_max(self) -> int | None:
        """The longest placeholder key the model call takes, or None for no limit."""
        return variable_key_max(self._model)

    @property
    def provider(self) -> str:
        """Who serves the model call, like ``slng``, or ``unknown``."""
        return provider_name(self._model)

    async def aclose(self) -> None:
        """Close the LLM, then the model call, when it has an ``aclose``."""
        await super().aclose()
        await close_model(self._model)

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        """Send one request through the model call and stream the reply as chunks."""
        return _ModelCallStream(
            self,
            node=self._node,
            model=self._model,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            settings=ModelSettings(tool_choice=tool_choice),
        )


class LiveKitModel(NativeModelCall):
    """A LiveKit LLM plugin as a ``ModelCall``: ``LiveKitModel(openai.LLM(model="gpt-4o"))``.

    Give it wherever a model goes: ``AgentSpec(model=...)``, ``summarizer=``, a task model,
    or ``HandoffContext(summarizer=...)``. Inside ``UnmuteAgent`` the plugin answers as the
    session's LLM does: the library fills the prompt, and the plugin's chunks pass through
    unchanged. Anywhere else, like inside ``FallbackModelCall``, ``stream`` sends the request
    through the plugin's ``chat`` and yields its text, tool calls and usage.

    Attributes:
        llm: The plugin. It keeps its own client, settings and retries.
    """

    def __init__(self, plugin: llm.LLM) -> None:
        """Wrap the plugin.

        Args:
            plugin: Any LiveKit LLM, ``llm.FallbackAdapter`` included.
        """
        super().__init__(self._chat)
        self.llm = plugin

    @property
    def model(self) -> str:
        """The model name the plugin sends."""
        return self.llm.model

    @property
    def provider(self) -> str:
        """Who serves the plugin's model, like ``openai``."""
        return self.llm.provider

    async def _chat(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Send a filled request through the plugin and yield its events.

        ``params`` reach an OpenAI-compatible plugin as ``extra_body``. A plugin takes no
        headers of its own per request, so ``headers`` are not sent. A tool call's provider
        data, like a Gemini thought signature, comes back as ``ToolCall.extra``, and a
        replayed call's ``extra_content`` goes out as LiveKit sends it.
        """
        chat_ctx = _chat_context(request.messages, [], llm.ChatContext.empty())
        tools: list[llm.Tool] = [
            llm.function_tool(_never_run, raw_schema=schema["function"]) for schema in request.tools
        ]
        usage: Usage | None = None
        async with self.llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            tool_choice=request.tool_choice or NOT_GIVEN,
            extra_kwargs={"extra_body": request.params} if request.params else NOT_GIVEN,
        ) as stream:
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = Usage(
                        chunk.usage.prompt_tokens,
                        chunk.usage.completion_tokens,
                        cached_input_tokens=chunk.usage.prompt_cached_tokens,
                        response_id=chunk.id or None,
                    )
                if chunk.delta is None:
                    continue
                if chunk.delta.content:
                    yield TextDelta(chunk.delta.content)
                for call in chunk.delta.tool_calls:
                    yield _tool_call(call)
        if usage is not None:
            yield usage


async def _never_run(raw_arguments: dict[str, Any]) -> None:
    """Stand in for a tool's handler: a ``ModelCall`` returns tool calls and runs none."""
    raise RuntimeError("a tool offered through LiveKitModel.stream is never run here")


class _ModelCallStream(llm.LLMStream):
    """One request of a ``ModelCallLLM``."""

    def __init__(
        self,
        owner: ModelCallLLM,
        *,
        node: "ModelNode",
        model: ModelCall,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        settings: ModelSettings,
    ) -> None:
        """Start the request. LiveKit runs ``_run`` in the background."""
        self._owner = owner
        self._node = node
        self._model = model
        self._settings = settings
        super().__init__(owner, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._retry_on_chunk_sent = False

    async def _run(self) -> None:
        """Stream the model call's events to LiveKit as chunks.

        A failed request reaches LiveKit as its own ``APIError``, so LiveKit retries what a
        retry can fix, and counts the rest. Anything else, like a prompt naming a value the
        state lacks, is raised as it is. A model call that retries itself, like a fallback
        chain or the router, makes its error final, so LiveKit does not walk the chain again.
        A request is never retried once text or a tool call went out: the caller would hear
        the reply twice. ``ModelNode.stream`` ends such a reply instead of raising.
        """
        try:
            async for chunk in self._node.stream(
                None, self.chat_ctx, self.tools, self._settings, prompt=None, site=self._owner.site
            ):
                self._event_ch.send_nowait(
                    chunk
                    if isinstance(chunk, llm.ChatChunk)
                    else llm.ChatChunk(
                        id=utils.shortuuid("unmute_"),
                        delta=llm.ChoiceDelta(role="assistant", content=chunk),
                    )
                )
        except APIError:
            raise
        except Exception as error:
            if (converted := livekit_error(error)) is None:
                raise
            if retries_itself(self._model):
                converted.retryable = False
            raise converted from error


def livekit_error(error: BaseException) -> APIError | None:
    """Return a model call's failure as LiveKit's own error, or None for any other error.

    An HTTP status becomes ``APIStatusError``, which LiveKit retries unless it is a 4xx
    no retry fixes. A timeout or a lost connection, here or behind an SDK error, becomes
    ``APITimeoutError`` or ``APIConnectionError``.
    """
    if (status := http_status(error)) is not None:
        return APIStatusError(str(error), status_code=status)
    match transport_failure(error):
        case "timeout":
            return APITimeoutError(str(error))
        case "connection":
            return APIConnectionError(str(error))
        case None:
            return None


def request_context(chat_ctx: llm.ChatContext, prompt: str | None) -> llm.ChatContext:
    """Return a copy of the request with the prompt as it reads now.

    The copy leaves out the running tool calls LiveKit adds to every request of the session
    (voice/agent_activity.py:3547). A step never made the delegate call that is still
    running. It reads that call as its own unfinished work and answers with nothing.

    Args:
        chat_ctx: The request LiveKit built.
        prompt: The agent's or step's prompt, placeholders still in it. None keeps the
            prompt the request holds.
    """
    ctx = chat_ctx.copy()
    running = {
        item.call_id
        for item in ctx.items
        if isinstance(item, llm.FunctionCall) and item.extra.get(RUNNING_PLACEHOLDER_KEY)
    }
    ctx.items[:] = [
        item
        for item in ctx.items
        if not (
            isinstance(item, llm.FunctionCall | llm.FunctionCallOutput) and item.call_id in running
        )
    ]
    index = ctx.index_by_id(INSTRUCTIONS_MESSAGE_ID)
    if index is not None and prompt is not None:
        ctx.items[index] = ctx.items[index].model_copy(update={"content": [prompt]})
    return ctx


def _prompt_messages(ctx: llm.ChatContext) -> list[tuple[int, dict[str, Any]]]:
    """Return each system and developer message with its place, in OpenAI format."""
    return [
        (index, {"role": item.role, "content": item.text_content or ""})
        for index, item in enumerate(ctx.items)
        if isinstance(item, llm.ChatMessage) and item.role in PROMPT_ROLES
    ]


class ModelNode:
    """Sends the requests of one agent and its steps, and traces each one."""

    def __init__(
        self,
        *,
        state: CallState | None,
        model: ModelCall | None,
        trace: SessionTrace | None,
        call: CallHooks | None = None,
        metrics_llm: llm.LLM | None = None,
        prompts_as_written: bool = False,
    ) -> None:
        """Keep what every request reads.

        Args:
            state: The call state the prompts read, or None.
            model: The model call, or None for the agent's or the session's LLM plugin. A
                ``LiveKitModel`` answers through its plugin, as the session's would.
            trace: The session's trace, or None for no spans.
            call: The call's hooks, or None. A request waits for the tools on events, and
                a main-agent request ends with their notes.
            metrics_llm: The LLM LiveKit holds for this node's requests, or None. Each model
                call request ends with one ``LLMMetrics`` on it, because no LiveKit
                ``LLMStream`` runs the request and so none reports it. A plugin's request
                reports itself, and so does one that ``ModelCallLLM.chat`` runs.
            prompts_as_written: True sends each prompt as written: a ``{{name}}`` in it is
                plain text, not a placeholder.
        """
        self.state = state
        self.model = model
        self.trace = trace
        self.call = call
        self.metrics_llm = metrics_llm
        self.prompts_as_written = prompts_as_written

    def adopt_trace(self, trace: SessionTrace) -> None:
        """Record every request into ``trace`` from now on, as a handoff's new agent does."""
        self.trace = trace

    async def stream(
        self,
        agent: Agent | None,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
        *,
        prompt: str | None,
        site: str,
        session: AgentSession | None = None,
        model: ModelCall | None = None,
        side: bool = False,
        kind: PromptKind = "agent",
        cut: list[Exception] | None = None,
    ) -> AsyncIterator[llm.ChatChunk | str]:
        """Send one request and stream the reply, as ``llm_node`` does.

        The call's hooks run in order: ``prompt`` on the request's prompt, still unfilled,
        then ``before_model``, ``wrap_model`` around a model call, and ``after_model``.

        Args:
            agent: The agent or step asking. None when a model call answers and no agent
                is at hand, as in ``ModelCallLLM.chat``.
            chat_ctx: The request LiveKit built.
            tools: The tools the model may call.
            model_settings: LiveKit's settings for this reply.
            prompt: The agent's or step's prompt now, placeholders still in it. None keeps
                the prompt the request holds.
            site: ``""`` for the main agent, ``task:<name>`` for a step.
            session: The call's session, for an agent no longer running: a decision the
                call's end asks for. None uses the agent's.
            model: Answers this request in place of the agent's model, like a summarizer.
            side: The request is not about the caller's newest line, like a summary. Its
                span joins the open turn.
            kind: Which prompt this is, for the ``prompt`` hooks. A side request and a
                finish check get no dev operation, as on Pipecat.
            cut: When given, gets the error of a reply that failed after its first text or
                tool call. Such a reply ends here with no error raised and no call failure:
                the caller already heard it. An error before that is raised as ever.

        A model call that fails for an agent fails the call through ``call``: LiveKit only
        logs a failed ``llm_node``. A plugin, a ``LiveKitModel`` or ``ModelCallLLM`` reports
        its own failure.

        Raises:
            MissingVariablesError: If the prompt names a value the state does not have.
            ValueError: If neither a model call nor an agent with an LLM can answer.
        """
        prepared = await self._prepare(
            agent, chat_ctx, tools, model_settings, prompt, site, session, model, side, kind
        )
        ctx, request = prepared.ctx, prepared.request
        prompts = _prompt_messages(ctx)
        as_written = self.prompts_as_written
        variables = template_variables(
            [message for _, message in prompts], self.state, site, as_written=as_written
        )
        filled = ctx if as_written else _filled(ctx, prompts, variables)
        telemetry = self.trace.telemetry if self.trace else None
        span = None
        if self.trace is not None and telemetry is not None:
            caller = None if side else prepared.callers.newest
            span = telemetry.start_model_request(
                model=prepared.name,
                provider=prepared.provider,
                site=site,
                messages=filled.to_provider_format("openai")[0],
                tools=request.tools,
                parent=self.trace.open_turn(caller, answering=False),
            )
        headers = telemetry.headers(span) if telemetry and span else {}
        source, label = _source(
            prepared, agent, session, filled, tools, model_settings, variables, headers
        )
        operation = (
            prepared.dev.request_started(
                name=label,
                model=prepared.name,
                provider=prepared.provider,
                caller=prepared.callers.newest,
            )
            if prepared.dev is not None and agent is not None
            else None
        )
        recorder = _ReplyRecorder(
            dev=prepared.dev,
            operation=operation,
            telemetry=telemetry,
            span=span,
            hooks=prepared.hooks,
            request=request,
            unmark=side_request() if prepared.unseen else None,
            answered=lambda: _answered(prepared.model, prepared.native, prepared.name),
            metrics_llm=None if prepared.native else self.metrics_llm,
            model=prepared.name,
            provider=prepared.provider,
        )
        try:
            async for raw in source:
                if (chunk := recorder.hear(raw)) is not None:
                    yield chunk
            if tail := recorder.tail():
                yield tail
            recorder.finished = True
        except Exception as exc:
            recorder.error = type(exc).__name__
            if not (recorder.began and not prepared.unseen):
                if self.call is not None and not prepared.native and agent is not None and not side:
                    self.call.fail_soon("llm", exc)
                raise
            _end_reply_early(exc, cut)
        finally:
            recorder.close()

    async def _prepare(
        self,
        agent: Agent | None,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
        prompt: str | None,
        site: str,
        session: AgentSession | None,
        model: ModelCall | None,
        side: bool,
        kind: PromptKind,
    ) -> "PreparedRequest":
        """Run the hooks before a request and return what it sends.

        Raises:
            ValueError: If neither a model call nor an agent with an LLM can answer.
        """
        model = model or self.model
        if model is None and agent is None:
            raise ValueError("A request with no model call needs the agent whose LLM answers.")
        plugin = model.llm if isinstance(model, LiveKitModel) else None
        native = model is None or plugin is not None
        callers = CallerHistory(chat_ctx)
        if self.call is not None and not side:
            newest = callers.newest
            await self.call.before_request(newest.text_content if newest else None)
        hooks = self.call.hooks if self.call is not None else None
        unseen = side or kind == "finish_check"
        dev = None if unseen else dev_observer(_running_session(agent, session))
        named = hooks or self.trace or dev or self.metrics_llm
        name, provider = _names(model, agent, session) if named else ("", "")
        ctx = request_context(chat_ctx, prompt)
        if self.call is not None and site == "" and not side:
            _add_notes(ctx, self.call)
        schemas = llm.ToolContext(tools).parse_function_tools("openai") if tools else []
        choice = tool_choice(model_settings.tool_choice)
        try:
            if hooks is not None and hooks.has("prompt"):
                _hook_prompt(ctx, hooks, PromptInfo(kind, site, name))
            wanted = hooks is not None and (hooks.has("before_model") or hooks.has("after_model"))
            request = ModelRequest(
                messages=ctx.to_provider_format("openai")[0] if not native or wanted else [],
                tools=schemas,
                site=site,
                tool_choice=choice,
                model=name,
            )
            if hooks is not None and hooks.has("before_model"):
                hooked = await hooks.run_before_model(request)
                if hooked.messages is not request.messages:
                    ctx = _chat_context(hooked.messages, request.messages, ctx)
                if native and (hooked.tools is not schemas or hooked.tool_choice != choice):
                    logger.warning("a before_model hook changed the tools: a native LLM ignores it")
                request = hooked
        except Exception as exc:
            if self.call is not None and agent is not None and not side:
                self.call.fail_soon("llm", exc)
            raise
        return PreparedRequest(
            model=model,
            plugin=plugin,
            native=native,
            callers=callers,
            hooks=hooks,
            unseen=unseen,
            dev=dev,
            name=name,
            provider=provider,
            ctx=ctx,
            request=request,
        )

    async def summarize(
        self, agent: Agent, source: llm.ChatContext, *, model: ModelCall | None = None
    ) -> str | None:
        """Ask for a summary of ``source`` for the agent taking over.

        Args:
            agent: The agent asking. Its LLM answers when no model call does.
            source: The chat to summarize.
            model: The summarizer. None uses the agent's own model.

        Returns:
            The summary, or None when it failed, came back empty or took too long.
        """
        request = summary_request(source.to_provider_format("openai")[0])
        ctx = llm.ChatContext(
            [llm.ChatMessage(role=m["role"], content=[m["content"]]) for m in request.messages]
        )
        text: list[str] = []

        async def ask() -> None:
            """Stream the summary and keep its text."""
            async for chunk in self.stream(
                agent,
                ctx,
                [],
                ModelSettings(),
                prompt=SUMMARY_PROMPT,
                site=SUMMARY_SITE,
                model=model,
                side=True,
                kind="summary",
            ):
                if isinstance(chunk, str):
                    text.append(chunk)
                elif chunk.delta is not None:
                    text.append(chunk.delta.content or "")

        try:
            await asyncio.wait_for(ask(), timeout=SUMMARY_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("the summary failed; the next agent hears what was said", exc_info=True)
            return None
        return "".join(text).strip() or None


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """One request after its hooks ran: what ``ModelNode.stream`` sends and records."""

    model: ModelCall | None
    plugin: llm.LLM | None
    native: bool
    callers: CallerHistory
    hooks: Hooks | None
    unseen: bool
    dev: DevObserver | None
    name: str
    provider: str
    ctx: llm.ChatContext
    request: ModelRequest


def _source(
    prepared: PreparedRequest,
    agent: Agent | None,
    session: AgentSession | None,
    filled: llm.ChatContext,
    tools: list[llm.Tool],
    model_settings: ModelSettings,
    variables: dict[str, str],
    headers: dict[str, str],
) -> tuple[AsyncIterator[llm.ChatChunk | str], str]:
    """Return the stream that answers a request, and the name the dev page shows for it.

    The session's or agent's LLM plugin answers a native request with the filled chat
    unchanged. A model call gets the request in OpenAI format with the values beside it.
    """
    request = prepared.request
    if prepared.native:
        plugin = prepared.plugin
        if plugin is None:
            assert agent is not None  # ponytail: stream checked this, for the type checker
            plugin = _native_llm(agent, session)
        source = _native(
            plugin,
            filled,
            tools,
            model_settings,
            request.params,
            session or (agent.session if agent else None),
        )
        return source, plugin.label
    model = prepared.model
    assert model is not None  # ponytail: native covers None, for the type checker
    sent = replace(
        request,
        messages=prepared.ctx.to_provider_format("openai")[0],
        variables=variables,
        headers=request.headers | headers,
    )
    options = _conn_options(agent, session) if agent is not None else None
    return _called(model, sent, prepared.hooks, options=options), type(model).__name__


class _ReplyRecorder:
    """Collects one reply as it streams, and reports it when it ends.

    Owns the words, the tool calls, the usage and the time to first byte. ``close`` ends
    the dev operation and the telemetry span and tells the ``after_model`` hooks.
    """

    def __init__(
        self,
        *,
        dev: DevObserver | None,
        operation: str | None,
        telemetry: Telemetry | None,
        span: Span | None,
        hooks: Hooks | None,
        request: ModelRequest,
        unmark: Callable[[], None] | None,
        answered: Callable[[], str | None],
        metrics_llm: llm.LLM | None = None,
        model: str = "",
        provider: str = "",
    ) -> None:
        """Start the clock.

        Args:
            dev: The dev observer, or None.
            operation: The dev operation of this request, or None.
            telemetry: The call's telemetry, or None.
            span: The request's ``llm`` span, or None.
            hooks: The call's hooks, or None.
            request: The request as sent, for the ``after_model`` hooks.
            unmark: Ends the side-request mark, or None.
            answered: Returns the name of the model call that answered, read after the request.
            metrics_llm: The LLM that gets this request's ``LLMMetrics``, or None for none.
            model: The model name the request went out with.
            provider: Who serves the model.
        """
        self.metrics_llm = metrics_llm
        self.model = model
        self.provider = provider
        self.ttft: float | None = None
        self.request_id = ""
        self.dev = dev if operation is not None else None
        self.operation = operation
        self.telemetry = telemetry
        self.span = span
        self.hooks = hooks
        self.request = request
        self.unmark = unmark
        self.answered = answered
        self.began = False
        self.started = time.perf_counter()
        self.ttfb: float | None = None
        self.text: list[str] = []
        self.calls: list[llm.FunctionToolCall] = []
        self.usage: llm.CompletionUsage | None = None
        self.error: str | None = None
        self.finished = False
        self.speech = ReasoningFilter()

    def hear(self, raw: llm.ChatChunk | str) -> llm.ChatChunk | str | None:
        """Return the chunk to pass on with its reasoning taken out, or None for no chunk."""
        if self.ttfb is None:
            self.ttfb = time.perf_counter() - self.started
        if (chunk := _spoken(raw, self.speech)) is not None:
            self.observe(chunk)
            self.began = self.began or _has_output(chunk)
            if self.ttft is None and _has_output(chunk):
                self.ttft = time.perf_counter() - self.started
        return chunk

    def observe(self, chunk: llm.ChatChunk | str) -> None:
        """Keep the words, tool calls and usage of one chunk, and show them to the dev page."""
        if isinstance(chunk, str):
            self.text.append(chunk)
        else:
            self.usage = chunk.usage or self.usage
            self.request_id = chunk.id or self.request_id
            if chunk.delta:
                self.text.append(chunk.delta.content or "")
                self.calls.extend(chunk.delta.tool_calls)
        if self.dev is not None and self.operation is not None:
            _dev_chunk(self.dev, self.operation, chunk)

    def tail(self) -> str | None:
        """Return the words the reasoning filter held back, recorded, or None."""
        if not (tail := self.speech.flush()):
            return None
        self.text.append(tail)
        self.began = True
        if self.dev is not None and self.operation is not None:
            self.dev.request_text(self.operation, tail)
        return tail

    def close(self) -> None:
        """Report the request's metrics, end its dev operation and span, tell the ``after_model`` hooks.

        The metrics go first, while the dev page still holds the request or its side mark, so
        it joins them to the operation and does not record the request twice.
        """
        answered = self.answered()
        self._report_metrics(answered)
        if self.unmark is not None:
            self.unmark()
        if self.dev is not None and self.operation is not None:
            self.dev.request_ended(
                self.operation,
                "ended" if self.finished else "failed" if self.error else "cancelled",
                first=self.ttfb,
                duration=time.perf_counter() - self.started,
                model=answered,
            )
        answer = ModelAnswer(
            "".join(self.text),
            tuple(_tool_call(call) for call in self.calls),
            Usage(
                self.usage.prompt_tokens,
                self.usage.completion_tokens,
                cached_input_tokens=self.usage.prompt_cached_tokens or None,
            )
            if self.usage
            else None,
        )
        interrupted = not self.finished and self.error is None
        if self.telemetry is not None and self.span is not None:
            if answered:
                self.span.set_attribute("gen_ai.response.model", answered)
            self.telemetry.end_model_request(
                self.span,
                text=answer.text or None,
                tool_calls=[call.to_openai() for call in answer.calls],
                input_tokens=answer.usage.input_tokens if answer.usage else None,
                output_tokens=answer.usage.output_tokens if answer.usage else None,
                cached_input_tokens=answer.usage.cached_input_tokens if answer.usage else None,
                ttfb=self.ttfb,
                error=self.error,
                interrupted=interrupted,
            )
        if self.hooks is not None and self.hooks.has("after_model"):
            response = ModelResponse(
                text=answer.text,
                tool_calls=answer.calls,
                error=self.error,
                interrupted=interrupted,
            )
            _after(self.hooks, self.request, response)

    def _report_metrics(self, answered: str | None) -> None:
        """Emit this request's ``LLMMetrics`` on the LLM, with LiveKit's own field rules.

        A request that produced no chunk or failed has nothing to report, as in LiveKit's
        ``LLMStream``. One that was cut after its first chunk reports as cancelled.
        """
        if self.metrics_llm is None or self.ttfb is None or self.error is not None:
            return
        duration = time.perf_counter() - self.started
        usage = self.usage
        self.metrics_llm.emit(
            "metrics_collected",
            LLMMetrics(
                timestamp=time.time(),
                request_id=self.request_id,
                ttft=self.ttft if self.ttft is not None else -1.0,
                duration=duration,
                cancelled=not self.finished,
                label=self.metrics_llm.label,
                completion_tokens=usage.completion_tokens if usage else 0,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                tokens_per_second=usage.completion_tokens / duration if usage and duration else 0.0,
                prompt_cached_tokens=usage.prompt_cached_tokens if usage else 0,
                metadata=Metadata(model_name=answered or self.model, model_provider=self.provider),
            ),
        )


def _add_notes(ctx: llm.ChatContext, call: CallHooks) -> None:
    """Add the notes of the tools on events to a main-agent request, before its values."""
    if (block := call.notes_block()) is None:
        return
    note = llm.ChatMessage(role="system", content=[block])
    last = ctx.items[-1] if ctx.items else None
    values = isinstance(last, llm.ChatMessage) and is_values(last.role, last.text_content)
    ctx.items.insert(len(ctx.items) - 1 if values else len(ctx.items), note)


def _running_session(agent: Agent | None, session: AgentSession | None) -> AgentSession | None:
    """Return ``session``, else the session the agent runs in, or None when it runs in none."""
    if session is not None or agent is None:
        return session
    try:
        return agent.session
    except RuntimeError:
        return None


def _dev_chunk(dev: DevObserver, operation: str, chunk: llm.ChatChunk | str) -> None:
    """Show the dev observer the words and the tool calls of one chunk."""
    if isinstance(chunk, str):
        dev.request_text(operation, chunk)
    elif chunk.delta is not None:
        dev.request_text(operation, chunk.delta.content or "")
        dev.request_calls(operation, (call.call_id for call in chunk.delta.tool_calls))


def _names(
    model: ModelCall | None, agent: Agent | None, session: AgentSession | None
) -> tuple[str, str]:
    """Return the model and provider names for the ``llm`` span."""
    if model is not None:
        return model_name(model), provider_name(model)
    assert agent is not None  # ponytail: stream checks this first
    native = _native_llm(agent, session)
    return native.model, native.provider


def _has_output(chunk: llm.ChatChunk | str) -> bool:
    """Return True when the chunk holds words or a tool call: what a caller can hear."""
    if isinstance(chunk, str):
        return bool(chunk)
    return chunk.delta is not None and bool(chunk.delta.content or chunk.delta.tool_calls)


def _answered(model: ModelCall | None, native: bool, asked: str) -> str | None:
    """Return the name of the model call that answered, read after the request, or None.

    A fallback chain names the call that answered last, so after a failover this differs
    from the name the request went out with. A native request has one model.
    """
    if model is None or native or not asked:
        return None
    name = answered_model(model)
    return None if name == "unknown" else name


def _conn_options(agent: Agent, session: AgentSession | None) -> APIConnectOptions:
    """Return the retry settings of the session the agent runs in, or LiveKit's defaults."""
    running = _running_session(agent, session)
    return running.conn_options.llm_conn_options if running else DEFAULT_API_CONNECT_OPTIONS


def _end_reply_early(error: Exception, cut: list[Exception] | None) -> None:
    """End a reply that failed after it began as it stands, and hand its error to ``cut``.

    The caller already heard the start, so the call goes on and nothing fails.
    """
    logger.warning(
        "the model failed after the reply began; the reply ends as it is", exc_info=error
    )
    if cut is not None:
        cut.append(error)


async def _called(
    model: ModelCall,
    request: ModelRequest,
    hooks: Hooks | None,
    *,
    options: APIConnectOptions | None = None,
) -> AsyncIterator[llm.ChatChunk | str]:
    """Stream the request through the ``ModelCall`` and the ``wrap_model`` hooks, as chunks.

    With ``options``, a request that fails before its first event is sent again as LiveKit
    would send it for its own LLM: up to ``max_retry`` more times, after the retry interval,
    when the error is one a retry can fix. A model call that retries itself is not retried,
    and neither is a request after its first event.
    """
    attempts = 1 if options is None or retries_itself(model) else options.max_retry + 1
    for attempt in range(attempts):
        started = False
        try:
            async for chunk in _chunks(model, request, hooks):
                started = True
                yield chunk
            return
        except Exception as error:
            converted = livekit_error(error)
            if started or attempt + 1 == attempts or converted is None or not converted.retryable:
                raise
            assert options is not None  # ponytail: attempts is 1 without options
            interval = options._interval_for_retry(attempt)
            logger.warning(
                "the model failed: %s; retrying in %ss (attempt %d of %d)",
                error,
                interval,
                attempt + 1,
                attempts - 1,
            )
            await asyncio.sleep(interval)


async def _chunks(
    model: ModelCall, request: ModelRequest, hooks: Hooks | None
) -> AsyncIterator[llm.ChatChunk | str]:
    """Stream one attempt of the request through the model call, as chunks."""
    events = hooks.run_model(request, model.stream) if hooks is not None else model.stream(request)
    async for event in events:
        if isinstance(event, TextDelta):
            yield event.text
            continue
        if isinstance(event, Usage):
            usage = llm.CompletionUsage(
                completion_tokens=event.output_tokens,
                prompt_tokens=event.input_tokens,
                prompt_cached_tokens=event.cached_input_tokens or 0,
                total_tokens=event.input_tokens + event.output_tokens,
            )
            yield llm.ChatChunk(id=event.response_id or utils.shortuuid("unmute_"), usage=usage)
            continue
        call = llm.FunctionToolCall(
            name=event.name,
            arguments=event.arguments,
            call_id=event.id,
            extra=dict(event.extra) or None,
        )
        yield llm.ChatChunk(
            id=utils.shortuuid("unmute_"),
            delta=llm.ChoiceDelta(role="assistant", tool_calls=[call]),
        )


def _hook_prompt(ctx: llm.ChatContext, hooks: Hooks, info: PromptInfo) -> None:
    """Run the ``prompt`` hooks on the request's prompt: the agent's, else its first system one."""
    index = ctx.index_by_id(INSTRUCTIONS_MESSAGE_ID)
    if index is None:
        index = next(
            (
                place
                for place, item in enumerate(ctx.items)
                if isinstance(item, llm.ChatMessage) and item.role in PROMPT_ROLES
            ),
            None,
        )
    item = ctx.items[index] if index is not None else None
    if index is None or not isinstance(item, llm.ChatMessage):
        return
    text = item.text_content or ""
    if (hooked := hooks.run_prompt(text, info)) != text:
        ctx.items[index] = item.model_copy(update={"content": [hooked]})


def _chat_context(
    messages: list[dict[str, Any]], sent: list[dict[str, Any]], source: llm.ChatContext
) -> llm.ChatContext:
    """Return the chat a ``before_model`` hook left, as LiveKit items.

    A message the hook kept as it was keeps its original item, with its id, images and
    data. Tool calls and their outputs keep theirs, found by call id. Only a message the
    hook wrote or changed is built anew, as text.

    Args:
        messages: The messages the hook returned.
        sent: The messages the hook was given.
        source: The chat they were made from.
    """
    calls = {item.call_id: item for item in source.items if isinstance(item, llm.FunctionCall)}
    outputs = {
        item.call_id: item for item in source.items if isinstance(item, llm.FunctionCallOutput)
    }
    kept = {
        _key(llm.ChatContext([item]).to_provider_format("openai")[0][0]): item
        for item in source.items
        if isinstance(item, llm.ChatMessage)
    }
    unchanged = {_key(message) for message in sent}
    items: list[llm.ChatItem] = []
    for message in messages:
        role, text = message.get("role", "user"), _text_of(message.get("content"))
        if role == "tool":
            call_id = str(message.get("tool_call_id", ""))
            items.append(
                outputs.get(call_id)
                or llm.FunctionCallOutput(call_id=call_id, name="", output=text, is_error=False)
            )
            continue
        key = _key({k: v for k, v in message.items() if k != "tool_calls"})
        if key in unchanged and key in kept:
            items.append(kept[key])
        elif text or not message.get("tool_calls"):
            items.append(llm.ChatMessage(role=cast(llm.ChatRole, role), content=[text]))
        for part in message.get("tool_calls") or []:
            function = part.get("function", {})
            items.append(
                calls.get(part.get("id"))
                or llm.FunctionCall(
                    call_id=part.get("id", ""),
                    name=function.get("name", ""),
                    arguments=function.get("arguments", ""),
                    extra=dict(part.get(EXTRA_CONTENT) or {}),
                )
            )
    return llm.ChatContext(items)


def _key(message: dict[str, Any]) -> str:
    """Return a message as stable text, to find it again."""
    return json.dumps(message, sort_keys=True, default=str)


def _text_of(content: object) -> str:
    """Return a message's text, from a string or a list of text parts."""
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def _after(hooks: Hooks, request: ModelRequest, response: ModelResponse) -> None:
    """Tell the ``after_model`` hooks in a task of their own, so a reply cut off waits for none.

    ponytail: a task, not an await, because the stream may be closing on a cancel.
    """
    task = asyncio.ensure_future(hooks.run_after_model(request, response))
    _AFTER.add(task)
    task.add_done_callback(_AFTER.discard)


_AFTER: set[asyncio.Task[None]] = set()
"""The ``after_model`` runs still going, kept until they end."""


def _native_llm(agent: Agent, session: AgentSession | None = None) -> llm.LLM:
    """Return the LLM plugin of the agent, or of its session: ``session``, else the agent's.

    Raises:
        RuntimeError: If neither has one.
    """
    model = agent.llm if isinstance(agent.llm, llm.LLM) else (session or agent.session).llm
    if not isinstance(model, llm.LLM):
        raise RuntimeError(
            "UnmuteAgent has no model: give AgentSession an llm=, or AgentSpec a model=."
        )
    return model


def _filled(
    ctx: llm.ChatContext, prompts: list[tuple[int, dict[str, Any]]], variables: dict[str, str]
) -> llm.ChatContext:
    """Return a copy of the request with its prompts filled: what the model reads."""
    filled = ctx.copy()
    texts = fill_messages([message for _, message in prompts], variables)
    for (index, original), message in zip(prompts, texts, strict=True):
        if message["content"] != original["content"]:
            filled.items[index] = filled.items[index].model_copy(
                update={"content": [message["content"]]}
            )
    return filled


def _spoken(chunk: llm.ChatChunk | str, speech: ReasoningFilter) -> llm.ChatChunk | str | None:
    """Return a chunk with its reasoning tags taken out, or None when nothing is left."""
    if isinstance(chunk, str):
        return speech.feed(chunk) or None
    if chunk.delta is None or not chunk.delta.content:
        return chunk
    content = speech.feed(chunk.delta.content)
    if not content and not chunk.delta.tool_calls and chunk.usage is None:
        return None
    return chunk.model_copy(
        update={"delta": chunk.delta.model_copy(update={"content": content or None})}
    )


def _tool_call(call: llm.FunctionToolCall) -> ToolCall:
    """Return a tool call LiveKit's plugin made, as the core's ``ToolCall``."""
    return ToolCall(call.call_id, call.name, call.arguments, call.extra or {})


async def _native(
    plugin: llm.LLM,
    ctx: llm.ChatContext,
    tools: list[llm.Tool],
    model_settings: ModelSettings,
    params: dict[str, Any] | None,
    session: AgentSession | None,
) -> AsyncIterator[llm.ChatChunk | str]:
    """Stream the LLM plugin's own chunks for the filled request, unchanged.

    ``params`` reach an OpenAI-compatible plugin as ``extra_body``. The session, when there
    is one, gives the connection options.
    """
    async with plugin.chat(
        chat_ctx=ctx,
        tools=tools,
        tool_choice=model_settings.tool_choice,
        conn_options=session.conn_options.llm_conn_options
        if session is not None
        else DEFAULT_API_CONNECT_OPTIONS,
        extra_kwargs={"extra_body": params} if params else NOT_GIVEN,
    ) as stream:
        async for chunk in stream:
            yield chunk


def metrics_llm(model_llm: NotGivenOr[llm.LLM]) -> "ModelCallLLM | None":
    """Return the LLM that gets the metrics of a model call's requests, or None.

    Only a ``ModelCallLLM`` does: a plugin's own stream reports its requests.
    """
    return model_llm if isinstance(model_llm, ModelCallLLM) else None


def model_parts(model: ModelCall | None) -> tuple[ModelCall | None, NotGivenOr[llm.LLM]]:
    """Return the model call an agent's requests go through, and the LLM LiveKit holds for it.

    A ``LiveKitModel`` answers through its plugin, and LiveKit holds the plugin itself. Any
    other model call is held as a ``ModelCallLLM``. None leaves both to the session.
    """
    if model is None:
        return None, NOT_GIVEN
    if isinstance(model, LiveKitModel):
        return model, model.llm
    return model, ModelCallLLM(model)


async def filled_llm(
    agent: Agent,
    chat_ctx: llm.ChatContext,
    tools: list[llm.Tool],
    model_settings: ModelSettings,
    *,
    state: CallState | None,
    call: CallHooks | None = None,
    save_values: bool = False,
) -> AsyncIterator[llm.ChatChunk | str]:
    """Run the agent's own LLM plugin with its prompt filled from the call state.

    Use it from the ``llm_node`` of a plain ``Agent``. Each request reads the values saved
    so far. With ``call``, the request also waits for the background prefetch, ends with
    the call's notes, and runs the call's ``prompt``, ``before_model`` and ``after_model``
    hooks. It needs no ``Telemetry``: ``traced_llm`` does the same and adds the spans.

    Args:
        agent: The agent whose session LLM answers.
        chat_ctx: The chat LiveKit sends.
        tools: The tools LiveKit offers.
        model_settings: LiveKit's settings for this reply.
        state: The call state the prompt's ``{{placeholders}}`` read.
        call: The call's hooks, or None.
        save_values: End the request with the current values, for ``values_tool``.
    """
    if save_values and state is not None and state.names:
        chat_ctx = chat_ctx.copy()
        chat_ctx.add_message(role="system", content=values_block(state))
    node = ModelNode(state=state, model=None, trace=None, call=call)
    async for chunk in node.stream(agent, chat_ctx, tools, model_settings, prompt=None, site=""):
        yield chunk


async def traced_llm(
    agent: Agent,
    trace: SessionTrace,
    chat_ctx: llm.ChatContext,
    tools: list[llm.Tool],
    model_settings: ModelSettings,
) -> AsyncIterator[llm.ChatChunk | str]:
    """Run the agent's own LLM plugin and record each request as an ``llm`` span.

    Use it from the ``llm_node`` of a plain ``Agent``. The plugin keeps its own client and
    retries. The prompt's ``{{placeholders}}`` are filled from the trace's state.

    Args:
        agent: The agent whose session LLM answers.
        trace: The session's trace, from ``trace_session``.
        chat_ctx: The chat LiveKit sends.
        tools: The tools LiveKit offers.
        model_settings: LiveKit's settings for this reply.
    """
    node = ModelNode(state=trace.state, model=None, trace=trace, call=trace.call)
    async for chunk in node.stream(agent, chat_ctx, tools, model_settings, prompt=None, site=""):
        yield chunk
