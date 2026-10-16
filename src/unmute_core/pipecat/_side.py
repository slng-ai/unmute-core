"""Prompts and the model requests a call makes outside the pipeline.

A finish check and a summary ask the model on their own, so no frame shows them.
``SideRequests`` renders prompts for the pipeline too: through the call's ``prompt``
hooks, then filled for a native service.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

from opentelemetry.trace import Span
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pydantic.dataclasses import dataclass as pydantic_dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.call import CallHooks
from unmute_core.handoff import SUMMARY_SITE, SUMMARY_TIMEOUT_SECONDS, summary_request
from unmute_core.hooks import Hooks, ModelResponse, PromptInfo, PromptKind
from unmute_core.model import (
    MissingVariablesError,
    ModelAnswer,
    ModelCall,
    ModelRequest,
    ToolCall,
    ToolChoice,
    Usage,
    collect,
    fill_messages,
    model_name,
    template_variables,
)
from unmute_core.pipecat._history import without_instructions
from unmute_core.pipecat._model import ModelCallLLMService
from unmute_core.pipecat._requests import RequestHooks, service_model
from unmute_core.state import CallState
from unmute_core.tasks import FINISH_TOOL_NAME, FinishCheck, StepRun, decided_finish

logger = logging.getLogger(__name__)


@original_errors
@pydantic_dataclass(frozen=True, config=VALUE)
class RequestSpans:
    """How a model request no frame shows is traced, in the turn it belongs to.

    ``TelemetryObserver`` sets it on ``UnmuteFlows``: only it knows the open turn.

    Attributes:
        start: Opens an ``llm`` span from the site, the messages sent and the tools.
        end: Closes it with the tool calls the model chose, in OpenAI format, the text it
            wrote, or None, and its token counts, when known.
    """

    start: Callable[[str, list[dict[str, Any]], list[dict[str, Any]]], Span]
    end: Callable[[Span, list[dict[str, Any]], str | None, Usage | None], None]


class SideRequests:
    """Renders prompts, and asks the model for finish checks and summaries.

    Attributes:
        request_spans: Set by ``TelemetryObserver``: traces a finish check or a summary in
            the turn it belongs to.
        summarizer: The summary's model, of the agent that holds the call. A handoff sets
            the next agent's.
    """

    def __init__(
        self,
        llm: LLMService[Any],
        state: CallState,
        *,
        call: CallHooks | None,
        requests: RequestHooks | None,
        summarizer: ModelCall | None,
    ) -> None:
        """Hold the call's LLM service, state, hooks and the summary's model.

        Args:
            llm: The LLM service the call runs on.
            state: The call state, which fills a prompt's placeholders.
            call: The call's hooks, or None.
            requests: What a pipeline request borrowed from the service, to give back
                before a side request goes out on the service's own client.
            summarizer: Writes the summary a ``summary`` history asks for, and answers the
                finish check when the call's LLM cannot be asked. None asks the call's LLM,
                when it can be asked.
        """
        self._llm = llm
        self._state = state
        self._call = call
        self._requests = requests
        self.summarizer = summarizer
        self._routed = isinstance(llm, ModelCallLLMService)
        self._warned_summary = False
        self.request_spans: RequestSpans | None = None

    @property
    def can_ask(self) -> bool:
        """Whether the call's LLM can be asked outside the pipeline, for a check or a summary.

        A ``ModelCallLLMService`` and an OpenAI-compatible service can. Any other service
        has no finish check, and a summary needs a ``summarizer`` model.
        """
        return isinstance(self._llm, ModelCallLLMService | BaseOpenAILLMService)

    def render(self, text: str, site: str, kind: PromptKind = "agent") -> str:
        """Return a prompt for the LLM after the ``prompt`` hooks, filled unless routed."""
        return self.fill(self.hooked(text, site, kind), site)

    def hooked(self, text: str, site: str, kind: PromptKind, model: str | None = None) -> str:
        """Return a prompt after the call's ``prompt`` hooks, still unfilled."""
        if self._call is None or not self._call.hooks.has("prompt"):
            return text
        info = PromptInfo(kind, site, model or service_model(self._llm))
        return self._call.hooks.run_prompt(text, info)

    def fill(self, text: str, site: str) -> str:
        """Return a prompt filled for a native service, as it is when routed."""
        if self._routed:
            return text
        messages = [{"role": "system", "content": text}]
        return fill_messages(messages, template_variables(messages, self._state, site))[0][
            "content"
        ]

    async def check(
        self, step: StepRun, check: FinishCheck, said: str, request: list[Any]
    ) -> tuple[str, dict[str, Any]] | None:
        """Run a finish check: the step's chat and the reply just said, finish or not_yet.

        Through a ``ModelCallLLMService`` the check is a ``ModelRequest``. A native
        OpenAI-compatible service gets it on its own client. Any other service asks the
        ``summarizer`` model. With none, there is no check, and the step goes on as the
        model leaves it.

        Returns:
            The finish's call id and arguments, or None.
        """
        site = step.site
        prompt = self.hooked(step.instructions() + "\n\n" + check.instruction, site, "finish_check")
        said_messages = [{"role": "assistant", "content": said}] if said.strip() else []
        chat = [*without_instructions(request), *said_messages]
        tools = [{"type": "function", "function": tool} for tool in check.tools]
        spans = self.request_spans
        span = spans.start(site, self._sent(prompt, site, chat), tools) if spans else None
        answer = ModelAnswer()
        try:
            async with asyncio.timeout(check.timeout_s):
                messages = [{"role": "system", "content": prompt}, *chat]
                model = None if self.can_ask else self.summarizer
                answer = await self.ask(messages, site, tools, "required", model)
        except Exception:
            logger.warning("finish check of %s failed; the step goes on", site, exc_info=True)
        calls = answer.calls
        if spans is not None and span is not None:
            spans.end(span, [call.to_openai() for call in calls], None, answer.usage)
        arguments = decided_finish((call.name, call.arguments) for call in calls)
        if arguments is None:
            return None
        call_id = next(call.id for call in calls if call.name == FINISH_TOOL_NAME)
        logger.info("finish check of %s: the runtime sends its finish", site)
        return call_id, arguments

    def _sent(self, prompt: str, site: str, chat: list[Any]) -> list[dict[str, Any]]:
        """Return what a finish check sends the model, its prompt filled, for the trace."""
        messages = [{"role": "system", "content": prompt}]
        with contextlib.suppress(MissingVariablesError):
            messages = fill_messages(messages, template_variables(messages, self._state, site))
        return [*messages, *(m for m in chat if isinstance(m, dict))]

    async def summary(self, messages: list[Any], summarizer: ModelCall | None) -> str | None:
        """Ask for a summary of ``messages``, traced as a request no frame shows.

        Returns:
            The summary, or None when there is no model to ask, or it failed, came back
            empty or took too long.
        """
        model = summarizer or self.summarizer
        if model is None and not self.can_ask:
            self._warn_no_summary()
            return None
        request = summary_request(messages)
        name = model_name(model) if model is not None else None
        system = request.messages[0]
        prompt = self.hooked(system["content"], SUMMARY_SITE, "summary", name)
        request = replace(request, messages=[{**system, "content": prompt}, *request.messages[1:]])
        spans = self.request_spans
        span = spans.start(SUMMARY_SITE, request.messages, []) if spans else None
        answer = ModelAnswer()
        try:
            async with asyncio.timeout(SUMMARY_TIMEOUT_SECONDS):
                answer = await self.ask(request.messages, SUMMARY_SITE, [], None, model)
        except Exception:
            logger.warning("the summary failed; the next agent hears what was said", exc_info=True)
        finally:
            if spans is not None and span is not None:
                spans.end(span, [], answer.text or None, answer.usage)
        return answer.text.strip() or None

    def _warn_no_summary(self) -> None:
        """Log once that a summary has no model to write it, so the next agent hears the chat."""
        if not self._warned_summary:
            self._warned_summary = True
            logger.warning(
                "a summary handoff has no summarizer and the call's %s cannot be asked, so "
                "the next agent hears the chat. Give the call a summarizer model.",
                type(self._llm).__name__,
            )

    async def ask(
        self,
        messages: list[Any],
        site: str,
        tools: list[dict[str, Any]],
        choice: ToolChoice | None,
        model: ModelCall | None = None,
    ) -> ModelAnswer:
        """Ask the model outside the pipeline, like a finish check or a summary.

        ``model`` answers when given, then a ``ModelCallLLMService``'s model call. A native
        OpenAI-compatible service gets the request on its own client. Any other service is
        not asked.

        With a ``ModelCallLLMService``, each replayed tool call gets back the provider data
        it streamed with, like a Gemini thought signature, and the provider data of each
        call the model makes here is kept the same way: a finish check's finish goes into
        the step's chat, and its next request replays it.

        Args:
            messages: The prompt first, then the chat.
            site: The prompt site.
            tools: The tools offered, in OpenAI format.
            choice: The tool choice, or None.
            model: Answers in place of the call's LLM.

        Returns:
            The text the model wrote, its tool calls and its usage.
        """
        extras = self._llm.tool_extras if isinstance(self._llm, ModelCallLLMService) else None
        if extras is not None:
            messages = extras.restore(messages)
        if model is None and isinstance(self._llm, ModelCallLLMService):
            model = self._llm.model_call
        hooks = self._call.hooks if self._call is not None else None
        request = ModelRequest(
            messages=messages,
            tools=tools,
            site=site,
            tool_choice=choice,
            model=model_name(model) if model is not None else service_model(self._llm),
        )
        if hooks is not None:
            request = await hooks.run_before_model(request)
        answer = ModelAnswer()
        error: str | None = None
        try:
            if model is not None:
                answer = await _streamed(model, request, self._state, hooks)
                if extras is not None:
                    for call in answer.calls:
                        extras.keep(call)
            elif self.can_ask:
                answer = await self.native_ask(cast(BaseOpenAILLMService, self._llm), request)
            return answer
        except Exception as failure:
            error = type(failure).__name__
            raise
        finally:
            if hooks is not None and hooks.has("after_model"):
                response = ModelResponse(answer.text, answer.calls, error=error)
                await hooks.run_after_model(request, response)

    async def native_ask(self, llm: BaseOpenAILLMService, request: ModelRequest) -> ModelAnswer:
        """Ask an OpenAI-compatible service on its own client, its prompt filled here.

        What a pipeline request borrowed is given back first, so this request sends only its
        own ``params``. The first message is the prompt when it is a system message.
        """
        if self._requests is not None:
            await self._requests.give_back()
        messages = request.messages
        first = messages[0] if messages else {}
        prompted = first.get("role") == "system" and isinstance(first.get("content"), str)
        context = LLMContext(messages=cast(Any, messages[1:] if prompted else messages))
        params = llm.get_llm_adapter().get_llm_invocation_params(
            context,
            system_instruction=self.fill(first["content"], request.site) if prompted else None,
            convert_developer_to_user=not llm.supports_developer_role,
        )
        built = llm.build_chat_completion_params(cast(Any, params))
        built |= {"stream": False}
        if request.tools:
            built |= {"tools": request.tools, "tool_choice": request.tool_choice}
        if request.params:
            built |= {"extra_body": dict(built.get("extra_body") or {}) | request.params}
        built.pop("stream_options", None)
        response = await llm._client.chat.completions.create(**built)
        message = response.choices[0].message if response.choices else None
        usage = getattr(response, "usage", None)
        return ModelAnswer(
            message.content or "" if message else "",
            tuple(
                ToolCall(call.id, call.function.name, call.function.arguments)
                for call in (message.tool_calls or [] if message else [])
                if call.type == "function"
            ),
            Usage(usage.prompt_tokens, usage.completion_tokens) if usage else None,
        )


async def _streamed(
    model: ModelCall, request: ModelRequest, state: CallState, hooks: Hooks | None
) -> ModelAnswer:
    """Stream a request through a model call and the ``wrap_model`` hooks, and collect it."""
    sent = replace(request, variables=template_variables(request.messages, state, request.site))
    events = hooks.run_model(sent, model.stream) if hooks is not None else model.stream(sent)
    return await collect(events)
