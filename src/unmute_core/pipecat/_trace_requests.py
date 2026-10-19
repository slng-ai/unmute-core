"""The ``llm`` spans of one Pipecat call: each model request from its start to its end."""

import contextlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from opentelemetry.trace import Span
from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallFromLLM,
    LLMContextFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFBMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.tts_service import TTSService

from unmute_core.model import (
    MissingVariablesError,
    ToolCall,
    Usage,
    answered_model,
    fill_messages,
    model_name,
    provider_name,
    template_variables,
)
from unmute_core.pipecat._model import ModelCallLLMService
from unmute_core.pipecat._realtime import is_realtime
from unmute_core.pipecat._services import service_setting
from unmute_core.pipecat._tasks import UnmuteFlows
from unmute_core.telemetry import Telemetry


def _tools(context: LLMContext | None) -> list[dict[str, Any]]:
    """Return the tools the context offers, as OpenAI function schemas."""
    standard = getattr(getattr(context, "tools", None), "standard_tools", None) or []
    return [{"type": "function", "function": tool.to_default_dict()} for tool in standard]


def tool_call(call: FunctionCallFromLLM) -> dict[str, Any]:
    """Return a function call the model chose, in OpenAI format."""
    arguments = json.dumps(call.arguments, ensure_ascii=False)
    return ToolCall(call.tool_call_id, call.function_name, arguments).to_openai()


class RequestTracker:
    """Owns the open ``llm`` span of a call and what the model said, chose and cost in it."""

    def __init__(
        self,
        telemetry: Telemetry,
        *,
        model: str,
        provider: str,
        flows: UnmuteFlows | None,
        turn_span: Callable[[], Span],
        said: Callable[[str], None],
        side_turn: Callable[[], Span] | None = None,
    ) -> None:
        """Keep what a span needs from the call.

        Args:
            telemetry: The call's telemetry.
            model: The model name the spans carry until a handoff gives another.
            provider: Who serves the model.
            flows: The call's tasks, which tell each request's prompt site.
            turn_span: Returns the open turn, opening one when the agent speaks first.
            said: Called with the text of a reply that finished, to add to the turn's lines.
            side_turn: Returns the turn a side request's span belongs to. None is ``turn_span``.
        """
        self._telemetry = telemetry
        self._model = model
        self._provider = provider
        self._flows = flows
        self._turn_span = turn_span
        self._said = said
        self._side_turn = side_turn or turn_span
        self._span: Span | None = None
        self._text: list[str] = []
        self._calls: list[dict[str, Any]] = []
        self._cut = False
        self._error: str | None = None
        self._service: LLMService | None = None
        self._usage: Usage | None = None
        self._ttfb: float | None = None
        self._context: LLMContextFrame | None = None

    @property
    def is_open(self) -> bool:
        """Whether a request's span is open."""
        return self._span is not None

    @property
    def has_content(self) -> bool:
        """Whether the open request already got text or tool calls."""
        return bool(self._text or self._calls)

    def hold_context(self, frame: LLMContextFrame) -> None:
        """Keep the context the next request answers."""
        self._context = frame

    def add_text(self, text: str) -> None:
        """Add a piece of the model's reply."""
        self._text.append(text)

    def add_calls(self, calls: Sequence[FunctionCallFromLLM]) -> None:
        """Add the tool calls the model chose."""
        self._calls.extend(tool_call(call) for call in calls)

    def mark_cut(self) -> None:
        """Note that the open request was cut off."""
        self._cut = self._span is not None

    def end_reply(self) -> None:
        """Close the span as its service ends the reply, as cut when an interruption came."""
        self.end(interrupted=self._cut)

    def note_error(self, frame: ErrorFrame, source: FrameProcessor) -> None:
        """Note the error type of the open request, when its own service reported it.

        An error from another processor, like a TTS service, is not the request's.
        """
        if isinstance(source, LLMService) and frame.processor is source and self._span is not None:
            self._error = type(frame.exception).__name__ if frame.exception else "ModelError"

    def same_reply(self, service: LLMService) -> bool:
        """Return True for a realtime service's second start of the reply it already began.

        OpenAI's realtime service starts a reply as it asks for one, and again as the
        reply's message arrives (services/openai/realtime/llm.py:954-957, 1202).
        """
        return is_realtime(service) and self._span is not None and not self.has_content

    def start(self, service: LLMService) -> None:
        """Open an ``llm`` span for the reply that starts now, with the request it answers."""
        site = self._flows.site if self._flows is not None else ""
        pushed, self._context = self._context, None
        context = pushed.context if pushed is not None else None
        self.end(interrupted=True)
        model, provider = self._names(service)
        self._span = self._telemetry.start_model_request(
            model=model,
            provider=provider,
            site=site,
            messages=self._sent(service, context, site),
            tools=_tools(context),
            parent=self._turn_span(),
        )
        self._text, self._calls, self._usage, self._ttfb = [], [], None, None
        self._cut = False
        self._error = None
        self._service = service

    def end(self, *, interrupted: bool = False) -> None:
        """Close the ``llm`` span with its text, tokens and time to first token.

        Args:
            interrupted: True when the reply never finished: an interruption, a new request
                or the end of the call came first. Pipecat still ends a cancelled reply with
                ``LLMFullResponseEndFrame``. Its text was never said in full, so it is no line.
        """
        if self._span is None:
            return
        usage = self._usage
        text = "".join(self._text)
        if not interrupted:
            self._said(text)
        if (answered := self._answered_by(self._service)) is not None:
            self._span.set_attribute("gen_ai.response.model", answered)
        self._telemetry.end_model_request(
            self._span,
            text=text or None,
            tool_calls=self._calls,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            cached_input_tokens=usage.cached_input_tokens if usage else None,
            ttfb=self._ttfb,
            error=self._error,
            interrupted=interrupted,
        )
        self._span = None

    def start_side(
        self, site: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Span:
        """Open an ``llm`` span for a request no frame shows, like a finish check or a summary.

        A request the call's end makes, like the decision about a held save, runs after the
        last turn closed. Its span belongs to that turn, not to a new one.
        """
        model, provider = self._names(self._flows.llm if self._flows is not None else None)
        return self._telemetry.start_model_request(
            model=model,
            provider=provider,
            site=site,
            messages=messages,
            tools=tools,
            parent=self._side_turn(),
        )

    def end_side(
        self, span: Span, tool_calls: list[dict[str, Any]], text: str | None, usage: Usage | None
    ) -> None:
        """Close a side request's ``llm`` span with what the model chose and its token counts."""
        flows_llm = self._flows.llm if self._flows is not None else None
        if (answered := self._answered_by(flows_llm)) is not None:
            span.set_attribute("gen_ai.response.model", answered)
        self._telemetry.end_model_request(
            span,
            text=text,
            tool_calls=tool_calls,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            cached_input_tokens=usage.cached_input_tokens if usage else None,
        )

    def metrics(
        self,
        frame: MetricsFrame,
        source: FrameProcessor,
        note: Callable[[Literal["llm_ttft", "tts_ttfb"], float], None],
    ) -> None:
        """Read token use and time to first byte from a service's metrics.

        Args:
            frame: The metrics frame.
            source: The service that pushed it.
            note: Called with ``llm_ttft`` or ``tts_ttfb`` and the wait, for the turn's latency.
        """
        for data in frame.data:
            if isinstance(data, LLMUsageMetricsData):
                self._usage = Usage(
                    data.value.prompt_tokens,
                    data.value.completion_tokens,
                    cached_input_tokens=data.value.cache_read_input_tokens,
                    reasoning_tokens=data.value.reasoning_tokens,
                    total_tokens=data.value.total_tokens,
                )
            elif isinstance(data, TTFBMetricsData) and isinstance(source, LLMService):
                self._ttfb = data.value
                note("llm_ttft", data.value)
            elif isinstance(data, TTFBMetricsData) and isinstance(source, TTSService):
                note("tts_ttfb", data.value)

    def _answered_by(self, service: LLMService | None) -> str | None:
        """Return the model that answered the request now, or None when it is not named.

        A ``FallbackModelCall`` names the call that answered last, so each request reads it
        again at its end.
        """
        if isinstance(service, ModelCallLLMService) and (
            name := answered_model(service.model_call)
        ):
            return name if name != "unknown" else None
        return None

    def _names(self, service: LLMService | None) -> tuple[str, str]:
        """Return the model first asked for the request, and the provider of the agent.

        A ``ModelCallLLMService`` names the call it asks first, read before the answer: with
        a fallback chain that is not always the one that answers. Any other service has the
        agent's model name.
        """
        if self._flows is not None and (model := self._flows.model) is not None:
            model_id, provider = model_name(model), provider_name(model)
        else:
            model_id, provider = self._model, self._provider
        if (
            isinstance(service, ModelCallLLMService)
            and (asked := model_name(service.model_call)) != "unknown"
        ):
            model_id = asked
        return model_id, provider

    def _sent(
        self, service: LLMService, context: LLMContext | None, site: str
    ) -> list[Mapping[str, Any]]:
        """Return what the model reads: the system instruction filled, then the messages.

        A provider's own messages, like a Gemini thought signature, are left out. A
        ``ModelCallLLMService`` request shows as it went out, after the call's hooks.
        """
        if isinstance(service, ModelCallLLMService) and (sent_request := service.last_request):
            with contextlib.suppress(MissingVariablesError):
                return fill_messages(sent_request.messages, sent_request.variables)
            return sent_request.messages
        sent = context.get_messages() if context else []
        messages: list[Mapping[str, Any]] = [m for m in sent if isinstance(m, dict)]
        system = service_setting(service, "system_instruction")
        if system is None:
            return messages
        prompt = [{"role": "system", "content": system}]
        if self._flows is not None:
            with contextlib.suppress(MissingVariablesError):
                prompt = fill_messages(prompt, template_variables(prompt, self._flows.state, site))
        return [*prompt, *messages]
