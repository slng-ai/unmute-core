"""Coval's sink: spans wait for the call's Coval ID, then go out with Coval's own names.

Ported from unmute@d70247b internal/generate/templates/telemetry/telemetry.py.tmpl:442-703 (the
exporter and the conversation submit), livekit_v1/tracing_coval.py.tmpl:92-103,
pipecat_v1/tracing.py.tmpl:287-295 (the simulation ID lookup) and
livekit_v1/tracing_coval.py.tmpl:697-930 (the span names).

The shared span contract stays neutral. ``_coval_shape`` writes Coval's names on a copy of each
span just before it is exported, so no other sink sees them.
"""

import json
import logging
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import field
from typing import Annotated, Any, Self

import httpx
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.telemetry._metadata import MetadataOnly
from unmute_core.telemetry._settings import CovalSettings
from unmute_core.telemetry._shape import (
    ChatMessage,
    MappingExporter,
    assistant_reply,
    chat_messages,
    read_json_list,
    with_attributes,
)
from unmute_core.telemetry._sinks import Sink
from unmute_core.telemetry._telemetry import (
    CALL_SPAN,
    MODEL_SPAN,
    STT_SPAN,
    TOOL_SPAN,
    TTS_SPAN,
    TURN_SPAN,
)

logger = logging.getLogger(__name__)

COVAL_TRACES_ENDPOINT = "https://api.coval.dev/v1/traces"
COVAL_SUBMIT_ENDPOINT = "https://api.coval.dev/v1/conversations:submit"
COVAL_SIMULATION_HEADER = "X-Simulation-Id"
COVAL_CONVERSATION_HEADER = "X-Conversation-Id"
COVAL_EXPORT_TIMEOUT_SECONDS = 30
COVAL_SUBMIT_TIMEOUT_SECONDS = 5
MAX_HELD_SPANS = 1024
MAX_TEXT_CHARS = 1024
MAX_PROMPT_CHARS = 8192
SIMULATION_KEY = "coval.simulation_id"
SIMULATION_HEADER_SUFFIX = "x-coval-simulation-id"


def coval_simulation_id(values: Mapping[str, object] | None) -> str | None:
    """Return the Coval simulation ID in a mapping, or None.

    The ``coval.simulation_id`` key wins. Then any key that ends in ``x-coval-simulation-id``,
    in any case, like ``sip.h.x-coval-simulation-id``. An operator can map the SIP header to
    any attribute name, so the header's own name is the safe match. Empty values are skipped.

    Args:
        values: Participant attributes, dispatch metadata, SIP headers or a request body.
    """
    if not values:
        return None
    exact = values.get(SIMULATION_KEY)
    if exact:
        return str(exact)
    return next(
        (
            str(value)
            for key, value in values.items()
            if value and str(key).lower().endswith(SIMULATION_HEADER_SUFFIX)
        ),
        None,
    )


def log_no_simulation(log: logging.Logger, call: str = "call") -> None:
    """Log that no simulation named the call, so its trace is filed when the call ends.

    Args:
        log: The adapter's logger.
        call: What to call the call in the line, like ``browser call``.
    """
    log.info(
        "No Coval simulation placed this %s, so its trace is filed as a Coval "
        "conversation when the call ends",
        call,
    )


def log_activated(
    log: logging.Logger, simulation_id: str, method: str, call: str | None = None
) -> None:
    """Log that the sink is now bound to a simulation.

    Args:
        log: The adapter's logger.
        simulation_id: The simulation's ID.
        method: How the ID was found.
        call: What to call the call, like ``browser call``. None leaves it out.
    """
    if call is None:
        log.info("Coval tracing active for simulation %s via %s", simulation_id, method)
    else:
        log.info("Coval tracing active for simulation %s via %s (%s)", simulation_id, method, call)


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class _Correlation:
    """How a call is bound to Coval: the request header, the ID and how the ID was found."""

    header: str
    value: str
    method: str | None

    @property
    def key(self) -> str:
        """Return the span attribute that names the ID."""
        if self.header == COVAL_SIMULATION_HEADER:
            return "coval.simulation_id"
        return "coval.conversation_id"


def _clip(text: object, limit: int = MAX_TEXT_CHARS) -> str:
    """Return the text cut to ``limit`` characters, with an ellipsis when it was cut."""
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def _put(attributes: dict[str, Any], key: str, value: object) -> None:
    """Set an attribute unless the value is missing or empty."""
    if value is not None and value != "" and value != []:
        attributes[key] = value


def _positive(value: object) -> bool:
    """Return True for a number above zero. A zero latency stands for no measurement."""
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _kept_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Return the messages that fit the prompt budget: system ones, then the newest others."""
    kept = {i for i, message in enumerate(messages) if message.role == "system"}
    spent = sum(len(json.dumps(messages[i].as_dict(), default=str)) for i in kept)
    for i in reversed(range(len(messages))):
        if i in kept:
            continue
        size = len(json.dumps(messages[i].as_dict(), default=str))
        if kept and spent + size > MAX_PROMPT_CHARS:
            break
        kept.add(i)
        spent += size
    return [messages[i] for i in sorted(kept)]


class _ToolDefinition(BaseModel):
    """One tool in a ``gen_ai.tool.definitions`` text: an OpenAI function schema or a bare name."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    name: Any = None
    function: dict[str, Any] | None = None

    @property
    def tool_name(self) -> Any:
        """The tool's name, from its ``function`` when that is not empty."""
        return (self.function or {"name": self.name}).get("name")


_TOOLS = TypeAdapter(list[Annotated[_ToolDefinition | object, Field(union_mode="left_to_right")]])


def _tool_names(definitions: object) -> list[str]:
    """Return the sorted names in a ``gen_ai.tool.definitions`` text."""
    tools = read_json_list(_TOOLS, definitions)
    names = [tool.tool_name for tool in tools if isinstance(tool, _ToolDefinition)]
    return sorted(str(name) for name in names if name)


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class _CovalShape:
    """Writes Coval's names on the attributes of a copy of one span.

    Attributes:
        correlation: How the call is bound to Coval, or None while it is not.
    """

    correlation: _Correlation | None

    def shape(self, span: ReadableSpan) -> ReadableSpan:
        """Return a copy of an ended span with Coval's names, text cut to Coval's limits.

        Text keys move to the Coval name, so a large prompt is not sent twice. Other neutral
        keys stay. A ``metrics.ttfb`` that is zero or less is dropped: a streaming provider
        reports zero for no measurement, and Coval would average it as an instant answer.

        Args:
            span: The ended span as the call emitted it. It is not changed.
        """
        attributes = dict(span.attributes or {})
        handlers = {
            CALL_SPAN: self._call,
            TURN_SPAN: self._turn,
            STT_SPAN: self._stt,
            MODEL_SPAN: self._llm,
            TOOL_SPAN: self._tool,
            TTS_SPAN: self._tts,
        }
        handlers.get(span.name, self._other)(attributes)
        if span.name == TOOL_SPAN and span.start_time is not None and span.end_time is not None:
            attributes["tool.latency_ms"] = round((span.end_time - span.start_time) / 1e6, 3)
        if "metrics.ttfb" in attributes and not _positive(attributes["metrics.ttfb"]):
            del attributes["metrics.ttfb"]
        return with_attributes(span, attributes)

    def _call(self, attributes: dict[str, Any]) -> None:
        """Name the call's origin, framework and Coval ID on the conversation span."""
        _put(attributes, "coval.call.origin", attributes.get("unmute.call.origin"))
        framework = attributes.get("unmute.call.framework")
        _put(attributes, "agent.provider", framework)
        if framework == "livekit":
            attributes["coval.agent_type"] = "sip_inbound_voice"
        if self.correlation is not None:
            attributes[self.correlation.key] = self.correlation.value
            _put(attributes, "coval.correlation.method", self.correlation.method)
        for key in ("input.value", "output.value"):
            if key in attributes:
                attributes[key] = _clip(attributes[key], MAX_PROMPT_CHARS)

    def _turn(self, attributes: dict[str, Any]) -> None:
        """Move a turn's transcripts and latencies to Coval's names."""
        _put(attributes, "turn.index", attributes.get("turn.number"))
        user, agent = attributes.pop("input.value", None), attributes.pop("output.value", None)
        _put(attributes, "turn.user_transcript", _clip(user))
        _put(attributes, "turn.agent_transcript", _clip(agent))
        if "turn.interrupted" in attributes:
            attributes["turn.interrupted"] = int(bool(attributes["turn.interrupted"]))
        if _positive(attributes.get("turn.e2e_seconds")):
            attributes["metrics.e2e_latency"] = attributes["turn.e2e_seconds"]
        if _positive(attributes.get("turn.eou_seconds")):
            attributes["vad.end_of_turn_delay"] = attributes["turn.eou_seconds"]

    def _stt(self, attributes: dict[str, Any]) -> None:
        """Move a transcript to Coval's name and add the provider and the transcription delay."""
        _put(attributes, "transcript", _clip(attributes.pop("output.value", None)))
        _put(attributes, "stt.providerName", attributes.get("gen_ai.provider.name"))
        if isinstance(attributes.get("metrics.ttfb"), int | float):
            attributes["stt.transcription_delay"] = attributes["metrics.ttfb"]

    def _llm(self, attributes: dict[str, Any]) -> None:
        """Move a model request's prompt, tools and reply to Coval's names, cut to its budget."""
        messages = chat_messages(attributes.pop("input.value", None))
        if messages:
            system = "\n\n".join(message.text for message in messages if message.role == "system")
            kept = _kept_messages(messages)
            _put(attributes, "gen_ai.system_instructions", _clip(system, MAX_PROMPT_CHARS))
            attributes["input"] = _clip(
                json.dumps([message.as_dict() for message in kept], default=str),
                MAX_PROMPT_CHARS,
            )
            attributes["prompt.message_count"] = len(messages)
            attributes["prompt.messages_traced"] = len(kept)
        names = _tool_names(attributes.pop("gen_ai.tool.definitions", None))
        _put(attributes, "tools", names)
        if names:
            attributes["tool_count"] = len(names)
        output = attributes.pop("output.value", None)
        calls = output is not None and assistant_reply(output) is not None
        if output is not None and not calls:
            attributes["output"] = _clip(output)
            attributes["response.length"] = len(str(output))
        reasons = attributes.get("gen_ai.response.finish_reasons")
        attributes["llm.finish_reason"] = (
            str(reasons[0]) if reasons else "tool_calls" if calls else "stop"
        )
        attributes["llm.cancelled"] = int(bool(attributes.get("llm.interrupted")))

    def _tool(self, attributes: dict[str, Any]) -> None:
        """Move a tool's result to Coval's name. ``shape`` adds its latency."""
        result = attributes.pop("gen_ai.tool.call.result", None)
        if result is not None:
            attributes["tool.result"] = _clip(result)
            attributes["tool.result.length"] = len(str(result))
        for key in ("function.arguments", "gen_ai.tool.call.arguments"):
            if key in attributes:
                attributes[key] = _clip(attributes[key])

    def _tts(self, attributes: dict[str, Any]) -> None:
        """Move the spoken text to Coval's name and add its character count."""
        _put(attributes, "transcript", _clip(attributes.pop("input.value", None)))
        _put(attributes, "tts.characters_count", attributes.get("metrics.character_count"))

    def _other(self, attributes: dict[str, Any]) -> None:
        """Cut the text on a span Coval has no names for, like a task step or an event."""
        for key in ("input.value", "output.value"):
            if key in attributes:
                attributes[key] = _clip(attributes[key])


def _coval_shape(span: ReadableSpan, correlation: _Correlation | None) -> ReadableSpan:
    """Return a copy of an ended span with Coval's names, text cut to Coval's limits.

    Args:
        span: The ended span as the call emitted it. It is not changed.
        correlation: How the call is bound to Coval, or None while it is not.
    """
    return _CovalShape(correlation).shape(span)


def _transcript_of(value: object) -> list[dict[str, str]]:
    """Return the call span's transcript with consecutive lines of one speaker merged."""
    transcript: list[dict[str, str]] = []
    for line in chat_messages(value):
        role, text = line.role, str(line.content or "").strip()
        if role not in ("user", "assistant") or not text:
            continue
        if transcript and transcript[-1]["role"] == role:
            transcript[-1]["content"] += " " + text
        else:
            transcript.append({"role": role, "content": text})
    return transcript


class _SubmitRequest(BaseModel):
    """The body Coval's conversation submit takes."""

    model_config = ConfigDict(frozen=True)

    transcript: list[dict[str, str]]
    metadata: dict[str, str]
    tags: list[str]
    external_conversation_id: str | None = None


class _Filed(BaseModel):
    """The conversation Coval filed."""

    conversation_id: Annotated[str, Field(min_length=1)]


class _SubmitResponse(BaseModel):
    """What Coval's conversation submit answers: the conversation it filed."""

    conversation: _Filed


def _submit(
    api_key: str,
    transcript: Sequence[Mapping[str, str]],
    *,
    agent: str,
    session_id: str,
    origin: str | None,
    transport: httpx.BaseTransport | None = None,
) -> str | None:
    """Register a call with Coval as a conversation. Return its ID, or None on any failure.

    This is the route for a call no simulation placed. Coval needs the conversation to exist
    before spans can attach to it. The wait is ``COVAL_SUBMIT_TIMEOUT_SECONDS`` because this
    runs after the call, in the host's shutdown window. It never raises.

    Args:
        api_key: The Coval API key.
        transcript: The call's lines, each ``{"role", "content"}``.
        agent: The agent's name.
        session_id: The call's ID, sent as the external conversation ID.
        origin: How the caller reached the agent, or None.
        transport: A transport for the HTTP client. Tests pass a mock.
    """
    metadata = {"agent": agent, "source": "unmute", "session_id": session_id}
    if origin:
        metadata["origin"] = origin
    try:
        request = _SubmitRequest(
            transcript=[dict(line) for line in transcript],
            metadata=metadata,
            tags=[agent],
            external_conversation_id=session_id or None,
        )
        with httpx.Client(timeout=COVAL_SUBMIT_TIMEOUT_SECONDS, transport=transport) as client:
            response = client.post(
                COVAL_SUBMIT_ENDPOINT,
                json=request.model_dump(mode="json", exclude_none=True),
                headers={"x-api-key": api_key},
            )
            response.raise_for_status()
            filed = _SubmitResponse.model_validate_json(response.content)
    except httpx.TimeoutException:
        logger.warning(
            "Registering this call with Coval took longer than its %s s budget, "
            "so this call is not filed.",
            COVAL_SUBMIT_TIMEOUT_SECONDS,
        )
    except (httpx.HTTPError, ValueError, TypeError) as error:
        logger.warning("Could not register this call with Coval: %s", error)
    else:
        return filed.conversation.conversation_id
    return None


class _Activation(BaseModel):
    """The one ID ``Coval.activate`` binds a call to, and how it was found."""

    model_config = ConfigDict(frozen=True)

    simulation_id: str | None = None
    conversation_id: str | None = None
    method: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        """Refuse a call that names both IDs or neither."""
        if (self.simulation_id is None) == (self.conversation_id is None):
            raise ValueError(
                "Coval.activate() takes exactly one of simulation_id or conversation_id."
            )
        return self


class _CovalExporter(MappingExporter):
    """Holds spans until the call's Coval ID is known, then sends them shaped for Coval.

    The provider belongs to one call, so no span needs a mark that names its call.
    """

    def __init__(
        self, api_key: str | None, *, file_calls: bool = True, filtered: bool = False
    ) -> None:
        """Start with no ID and an empty buffer.

        Args:
            api_key: The Coval API key, or None.
            file_calls: Whether ``shutdown`` files a call that never got an ID.
            filtered: True when the sink sends metadata only, so there is no transcript to file.
        """
        super().__init__()
        self._api_key = api_key
        self._file_calls = file_calls
        self._filtered = filtered
        self._lock = threading.Lock()
        self._held: deque[ReadableSpan] = deque(maxlen=MAX_HELD_SPANS)
        self._dropped = 0
        self._correlation: _Correlation | None = None

    def shape(self, span: ReadableSpan) -> ReadableSpan:
        """Return the span with Coval's names, for the ID the call is bound to."""
        return _CovalShape(self._correlation).shape(span)

    def activate(self, header: str, correlation_id: str, method: str | None = None) -> None:
        """Bind the call to its ID and send the held spans."""
        if self._api_key is None:
            logger.warning("Coval has no API key, so this call's spans are dropped.")
            return
        delegate = OTLPSpanExporter(
            endpoint=COVAL_TRACES_ENDPOINT,
            headers={"x-api-key": self._api_key, header: correlation_id},
            timeout=COVAL_EXPORT_TIMEOUT_SECONDS,
        )
        correlation = _Correlation(header, correlation_id, method)
        with self._lock:
            if self._delegate is not None:
                raise RuntimeError("This call is already bound to a Coval ID.")
            self._delegate, self._correlation = delegate, correlation
            held, dropped = list(self._held), self._dropped
            self._held.clear()
        if dropped:
            logger.warning("Dropped %d span(s) before the Coval ID arrived.", dropped)
        if held:
            shape = _CovalShape(correlation)
            delegate.export([shape.shape(span) for span in held])

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Send the spans shaped for Coval, or hold them until the call has its ID."""
        with self._lock:
            delegate, correlation = self._delegate, self._correlation
            if delegate is None:
                self._dropped += max(0, len(self._held) + len(spans) - MAX_HELD_SPANS)
                self._held.extend(spans)
        if delegate is None:
            return SpanExportResult.SUCCESS
        shape = _CovalShape(correlation)
        return delegate.export([shape.shape(span) for span in spans])

    def shutdown(self) -> None:
        """Shut down. A call with no Coval ID yet is filed as a conversation first.

        The provider calls this after ``Telemetry.close`` flushed, so every span of the call is
        held here. The filing waits at most ``COVAL_SUBMIT_TIMEOUT_SECONDS`` for Coval.
        """
        if self._delegate is None:
            self._file_call()
        elif self._correlation is not None:
            logger.info("Coval already correlated this call as %s", self._correlation.value)
        super().shutdown()

    def _file_call(self) -> None:
        """Register the call with Coval from its transcript and send the held spans to it.

        Does nothing when no span is held. Logs why when it does not file: no key, ``file_calls``
        off, a filter, or no transcript.
        """
        held = list(self._held)
        if not held:
            return
        if self._api_key is None:
            logger.warning(
                "Coval has no API key, so this call is not filed and its %d held span(s) "
                "are dropped.",
                len(held),
            )
            return
        if not self._file_calls:
            logger.info(
                "Coval(file_calls=False): this call is not filed and its %d held span(s) "
                "are dropped.",
                len(held),
            )
            return
        if self._filtered:
            logger.info(
                "This Coval sink sends metadata only, so there is no transcript to file. "
                "%d held span(s) are dropped.",
                len(held),
            )
            return
        root = next((span for span in held if span.name == CALL_SPAN and span.parent is None), None)
        attributes = dict(root.attributes or {}) if root is not None else {}
        transcript = _transcript_of(attributes.get("input.value"))
        if not transcript:
            logger.warning(
                "This call produced no transcript, so there is nothing to register with "
                "Coval and its %d held span(s) are dropped.",
                len(held),
            )
            return
        origin = attributes.get("unmute.call.origin")
        conversation_id = _submit(
            self._api_key,
            transcript,
            agent=str(attributes.get("agent.name") or ""),
            session_id=str(attributes.get("session.id") or ""),
            origin=str(origin) if origin else None,
        )
        if conversation_id is None:
            return
        self.activate(COVAL_CONVERSATION_HEADER, conversation_id, "conversation_submit")
        logger.info(
            "Filed this call as Coval conversation %s with %d span(s)", conversation_id, len(held)
        )


@original_errors
@dataclass(kw_only=True, config=VALUE)
class Coval(Sink):
    """Coval's OTLP endpoint. Spans wait until the call has a simulation or conversation ID.

    A missing key is not an error, as in Unmute: an evaluation credential must not break a
    live call. Without a key, spans are held and then dropped.

    A call that no simulation placed is filed as a Coval conversation when the call ends,
    from its transcript, and its spans go out against that conversation. That needs an
    API key, ``file_calls`` on and no ``filter``, because metadata only never sends the
    transcript.

    Every Coval name is added to a copy of each span inside this sink. Other sinks receive the
    shared span contract unchanged.

    Attributes:
        api_key: The Coval API key, or None to export nothing.
        simulation_id: The simulation ID from ``COVAL_SIMULATION_ID``, for a local run. An
            adapter's ``correlate_coval`` reads it.
        file_calls: File a call no simulation placed as a Coval conversation at the end.
            Unmute does for every such call. Set it to False to file nothing.
    """

    api_key: str | None
    simulation_id: str | None = None
    file_calls: bool = True
    _exporter: _CovalExporter | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_settings(
        cls,
        settings: CovalSettings,
        *,
        filter: MetadataOnly | None = None,
        file_calls: bool = True,
    ) -> Self:
        """Build the sink from loaded settings. A missing key logs a warning.

        Args:
            settings: The loaded Coval settings.
            filter: Lets only metadata through when set, like ``MetadataOnly()``.
            file_calls: File a call no simulation placed as a Coval conversation at the end.
        """
        key = settings.api_key.get_secret_value() if settings.api_key else None
        logger.info(
            "Coval tracing on: traces to %s, COVAL_API_KEY %s",
            COVAL_TRACES_ENDPOINT,
            "present" if key else "MISSING",
        )
        if key is None:
            logger.warning("COVAL_API_KEY is not set, so no spans will reach Coval.")
        return cls(
            api_key=key,
            simulation_id=settings.simulation_id,
            filter=filter,
            file_calls=file_calls,
        )

    @classmethod
    def from_env(
        cls,
        *,
        filter: MetadataOnly | None = None,
        file_calls: bool = True,
    ) -> Self:
        """Read ``COVAL_API_KEY`` and ``COVAL_SIMULATION_ID``. A missing key logs a warning."""
        return cls.from_settings(CovalSettings(), filter=filter, file_calls=file_calls)

    def exporter(self) -> SpanExporter:
        """Build the call's exporter, which holds spans until ``activate``."""
        self._exporter = _CovalExporter(
            self.api_key, file_calls=self.file_calls, filtered=self.filter is not None
        )
        return self._exporter

    def activate(
        self,
        *,
        simulation_id: str | None = None,
        conversation_id: str | None = None,
        method: str | None = None,
    ) -> None:
        """Name the Coval run this call belongs to, and send the spans held so far.

        The held spans go out on the calling thread. From async code, call it with
        ``asyncio.to_thread``.

        Args:
            simulation_id: The ID of the Coval simulation that placed the call.
            conversation_id: The ID Coval returned when the call was submitted.
            method: How the ID was found, like ``dispatch_metadata``. It is written on the call
                span as ``coval.correlation.method``.

        Raises:
            ValueError: If not exactly one of the two IDs is given.
            RuntimeError: If no ``Telemetry`` was built with this sink yet, or the call is
                already bound to a Coval ID.
        """
        try:
            _Activation(simulation_id=simulation_id, conversation_id=conversation_id, method=method)
        except ValidationError as error:
            raise (error.errors()[0].get("ctx", {}).get("error") or error) from None
        if self._exporter is None:
            raise RuntimeError("Build the call's Telemetry with this Coval sink before activate().")
        if simulation_id is not None:
            self._exporter.activate(COVAL_SIMULATION_HEADER, simulation_id, method)
        elif conversation_id is not None:
            self._exporter.activate(COVAL_CONVERSATION_HEADER, conversation_id, method)
