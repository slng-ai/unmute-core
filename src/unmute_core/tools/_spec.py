"""What a tool is, how the model sees it, and how a call runs.

Locked arguments follow Unmute's ``inject`` (unmute@d0c67f1 internal/generate/
templates_lower.go injectExpr): the model never sees them, and the library fills them from
the call state. A locked value is sent as its parameter's JSON type, and the filled
arguments are checked against the whole schema. A value type is sent only as a parameter
type it fits: an ``int`` as an integer, a number or text, a date only as text.
"""

import asyncio
import copy
import inspect
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import field
from functools import cached_property
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from jsonschema.protocols import Validator
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    SkipValidation,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    model_validator,
)
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state import CallState, StateOrigin, StateRefused
from unmute_core.state._call_info import call_model
from unmute_core.state._paths import (
    PLACEHOLDER,
    ValuePathError,
    _model_class,
    is_list,
    is_single_value,
    literal_choices,
    parse_path,
    path_annotation,
    value_base,
)
from unmute_core.tools._announce import checked_lines
from unmute_core.tools._errors import ArgumentsMissing as ArgumentsMissing
from unmute_core.tools._errors import ToolError as ToolError
from unmute_core.tools._errors import ToolRefused as ToolRefused
from unmute_core.tools._fill import TRANSCRIPT_MAX_BYTES as TRANSCRIPT_MAX_BYTES
from unmute_core.tools._fill import Transcript as Transcript
from unmute_core.tools._fill import fill_text as fill_text
from unmute_core.tools._fill import locked_value as locked_value
from unmute_core.tools._fill import old_call_fact
from unmute_core.tools._fill import require_set as require_set
from unmute_core.tools._lines import (
    Assign,
    SpokenLines,
    ToolReply,
    match_words,
    read_path,
    result_paths,
    success_word,
)
from unmute_core.tools._schema import check_parameters, model_schema, model_shape
from unmute_core.tools._spoken import form_at

logger = logging.getLogger(__name__)

TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
ToolEffect = Literal["returns_data", "ends_conversation"]
"""What a tool does to the call: returns data for the model, or ends the call once done."""
ToolInterruption = Literal["provider_default", "cancel", "continue"]
"""What the caller's speech does to a running tool call."""
ToolSource = Literal["model", "event"]
"""Who called a tool: the model, or an event the tool runs on."""
ToolStarted = Callable[["ToolSpec"], Awaitable[None] | None]
"""Called once a tool call is checked, just before the tool runs. Its announce lines are
filled. An awaitable it returns is awaited before the tool runs."""
ToolEventSink = Callable[[str, dict[str, Any]], object]
"""Told each tool call's stage: ``tool.started``, ``tool.succeeded``, ``tool.failed`` or
``tool.refused``, with the tool's ``name`` and ``source`` (``model`` or ``event``). Names
and timing only, never argument values. ``tool.replied`` follows when code speaks a reply
line for the result, in place of a model reply."""
HIDDEN_RESULT = MappingProxyType({"ok": True})
"""What the model reads from a tool that succeeded and hides its result."""
SCALAR_TYPES = frozenset({"string", "integer", "number", "boolean"})
_FITS = MappingProxyType(
    {
        "int": frozenset({"integer", "number", "string"}),
        "float": frozenset({"number", "string"}),
        "bool": frozenset({"boolean", "string"}),
        "str": SCALAR_TYPES,
        "Id": SCALAR_TYPES,
    }
)
"""A value's type to the parameter types it can be sent as. Text with a format, like a date,
is sent only as a string; plain text and an id may read as a number or a boolean."""


def check_tool_names(sources: Sequence[tuple[str, Sequence[str]]]) -> None:
    """Refuse an agent that offers one tool name twice. Both adapters call it at build.

    Args:
        sources: Where the names come from, like ``("a native tool", ["end_call"])``.

    Raises:
        ValueError: The message names the tool and the sources that clash.
    """
    found: dict[str, list[str]] = {}
    for source, names in sources:
        for name in names:
            found.setdefault(name, []).append(source)
    for name, places in found.items():
        if len(places) > 1:
            raise ValueError(
                f"the tool '{name}' is offered twice on one agent: as {' and as '.join(places)}. "
                "Tool names must be unique. Rename one."
            )


def tool_succeeded(result: object) -> bool:
    """Return True when a tool answered rather than reported an error.

    A failure a tool caught reads as an ``error`` field that holds something, a ``refused``
    field that holds something, or ``ok: false``. ``"error": null`` is no error.
    """
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return True
    if isinstance(result, Mapping):
        return (
            not result.get("error") and not result.get("refused") and result.get("ok") is not False
        )
    return True


def replies_to(result: object, *, spoke: bool) -> bool:
    """Return whether the model replies to a tool's result.

    Not to a refusal for a missing argument when the reply that made the call spoke: it
    asked the caller already, so the caller answers next. A second reply would ask twice.

    Args:
        result: What the tool call returned.
        spoke: The reply that made the call said something.
    """
    return not (spoke and isinstance(result, Mapping) and bool(result.get("missing")))


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class ToolRun:
    """One tool call, as a tool hook sees it.

    Attributes:
        name: The tool.
        arguments: The arguments the model wrote, before the locked ones are filled. A
            tool on an event has none.
        source: ``model`` when the model called it, ``event`` for a tool on an event.
    """

    name: str
    arguments: SkipValidation[dict[str, Any]] = field(default_factory=dict)
    source: ToolSource = "model"


class _ToolEvent(BaseModel):
    """What a sink and an observer are told of a tool call: names and timing, no values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    source: ToolSource
    ms: int | None = None
    error: str | None = None


class ToolHooks(Protocol):
    """What runs around a tool call: ``unmute_core.hooks.Hooks`` or a call's ``CallHooks``."""

    async def run_before_tool(self, run: ToolRun) -> ToolRun:
        """Change the call, or raise ``ToolRefused`` to stop it."""
        ...

    async def run_tool(self, run: ToolRun, call: Callable[[ToolRun], Awaitable[object]]) -> object:
        """Run the call through the wrapping hooks."""
        ...

    async def run_after_tool(self, run: ToolRun, result: object) -> object:
        """Return the result the model reads."""
        ...

    def emit(self, event: str, data: object) -> None:
        """Tell the observers of a tool event."""
        ...


class ToolExecutor(Protocol):
    """Runs a tool once its arguments are checked and its locked values filled.

    This is the plug point for every way to run a tool. The library ships ``Webhook`` and
    ``LocalHandler``. Another package can add its own kinds. The ``ToolSpec`` that holds an
    executor names its output model, so the registry checks every result the same way.

    An executor that validates its arguments in Python may also define
    ``validate(arguments) -> dict``, returning them as the tool reads them, as JSON data.
    ``ToolRegistry.check`` then refuses a bad input before a step asks the caller to
    confirm it. One whose ``run`` takes a ``context`` keyword gets an event tool's event and
    transcript too.
    One with ``checks_extra_arguments = True`` gets arguments the schema does not name,
    so it can refuse them in its own words.
    """

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Run the tool and return its result for the model.

        Args:
            arguments: The model's arguments with the locked values filled in.
            state: The call state, or None for an agent without one.

        Returns:
            The result, as JSON data or an instance of the tool's output model.

        Raises:
            ToolError: With the text the model should hear.
        """
        ...


def _output_model(value: object) -> type[BaseModel]:
    """Pass a Pydantic model class: the shape of a tool's result."""
    if not (isinstance(value, type) and issubclass(value, BaseModel)):
        raise ValueError(
            f"output must be a Pydantic model class, like `class Order(BaseModel)`, got {value!r}"
        )
    return value


def _executes(value: object) -> object:
    """Pass an object whose ``run`` is a coroutine function, as a ``ToolExecutor``'s is."""
    if not inspect.iscoroutinefunction(getattr(value, "run", None)):
        raise ValueError(
            f"needs a ToolExecutor with an async run(arguments, state) method, got "
            f"{type(value).__name__}"
        )
    return value


@runtime_checkable
class ReadyChecking(Protocol):
    """An executor that refuses a call, before anything is said, while a value is unset."""

    def check_ready(self, state: CallState | None) -> None:
        """Raise ``ToolRefused`` naming the values the call still needs."""
        ...


@runtime_checkable
class Prechecking(Protocol):
    """An executor that may answer a call with an outcome before the tool runs."""

    def precheck(self, arguments: dict[str, Any]) -> BaseModel | Mapping[str, Any] | None:
        """Return the outcome that stands in for the tool's result, or None to run it."""
        ...


@runtime_checkable
class InputValidating(Protocol):
    """An executor that checks its arguments in Python, before a step asks to confirm them."""

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return the arguments as the tool reads them, as JSON data."""
        ...


@runtime_checkable
class TranscriptSource(Protocol):
    """Hooks that can read the call's transcript, like a call's ``CallHooks``."""

    def transcript(self) -> list[dict[str, Any]]:
        """Return the chat so far, as ``{"role", "content"}`` messages."""
        ...


@runtime_checkable
class Draining(Protocol):
    """An executor with sends still in flight that finish at the end of the call."""

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for the sends still in flight, at most ``timeout`` seconds."""
        ...


def _parameter_type(parameters: Mapping[str, Any], name: str) -> str | None:
    """Return a parameter's JSON type when it is one scalar type, else None."""
    declared = parameters.get("properties", {}).get(name, {}).get("type")
    return declared if isinstance(declared, str) and declared in SCALAR_TYPES else None


def _is_scalar(value: object, expected: str) -> bool:
    """Return True when a JSON value is of one scalar JSON type. A float must be finite."""
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if expected == "integer":
        return isinstance(value, int)
    return not isinstance(value, float) or math.isfinite(value)


def coerce_locked(value: object, expected: str | None) -> object:
    """Send a locked value as its parameter's JSON type.

    A number or a boolean fills a string as its JSON text. Text fills a number or a boolean
    when it reads as one exactly: ``"12"`` is 12, ``"1"`` is no boolean.

    Args:
        value: The value as the state holds it.
        expected: The parameter's JSON type, or None when it is not one scalar type.

    Returns:
        The value as that type.

    Raises:
        ToolError: If the value cannot be sent as that type.
    """
    if expected is None or value is None:
        return value
    if expected == "string" and not isinstance(value, str):
        return (
            json.dumps(value) if _is_scalar(value, "number") or isinstance(value, bool) else value
        )
    if expected != "string" and isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ToolError(f"{value!r} is not a valid {expected}") from None
    if not _is_scalar(value, expected):
        raise ToolError(f"{value!r} is not a valid {expected}")
    return value


def _coerces(value: object, expected: str) -> bool:
    """Return True when ``coerce_locked`` sends the value as that type."""
    try:
        coerce_locked(value, expected)
    except ToolError:
        return False
    return True


def sendable_as(annotation: object) -> frozenset[str]:
    """Return the parameter JSON types a State value can always be sent as.

    Every choice of a fixed set must read as the type. Text with a format, like a date, is
    sent only as a string.

    Args:
        annotation: The State value's type.

    Returns:
        The JSON types it can be sent as.
    """
    base = value_base(annotation)
    if base == "Literal":
        choices = literal_choices(annotation)
        return frozenset(kind for kind in SCALAR_TYPES if all(_coerces(c, kind) for c in choices))
    return _FITS.get(base, frozenset({"string"}))


def _described(text: str, info: ValidationInfo) -> str:
    """Give a blank description the tool's name: the model reads it, and the name says what it is."""
    return text if text.strip() else str(info.data.get("name", ""))


def _announce_lines(lines: str | Sequence[str], info: ValidationInfo) -> tuple[str, ...]:
    """Strip the announce lines, and refuse a placeholder that is not a value."""
    checked = checked_lines(lines, placeholders=True)
    for line in checked:
        if old := old_call_fact(line):
            raise ValueError(f"tool {info.data.get('name')}: announce line {line!r}: {old}")
        if _stray_braces(line):
            raise ValueError(
                f"tool {info.data.get('name')}: announce line {line!r} holds a placeholder that "
                "is not {{name}} or {{name.field}}"
            )
    return checked


def _stray_braces(text: str) -> bool:
    """Return True when text keeps ``{{`` or ``}}`` after its known placeholders are removed."""
    left = PLACEHOLDER.sub("", text)
    return "{{" in left or "}}" in left


@original_errors
@dataclass(frozen=True, config=VALUE)
class ToolSpec:
    """One tool: what the model sees, and what runs when it calls it.

    Attributes:
        name: Letters, digits, ``_`` and ``-``, at most 64. The model calls it by this name.
        description: What the tool does and when to call it, for the model.
        parameters: The JSON Schema of the arguments, an object at the root, in the subset
            ``validate_parameter_schema`` accepts.
        output: The Pydantic model of the result. A result that succeeded and does not fit
            it fails the call, and the model reads the validated result. A result that
            reports an error (``error``, ``refused``, ``ok: false``) is never checked.
        executor: What runs the call.
        locked: Argument name to its value: ``"{{state_name}}"``, text with placeholders, or a
            literal. The model never sees these arguments, and a value it sends is replaced.
        announce: A line said as the tool starts, so a slow tool is not silence: one
            line, or alternatives. ``{{name}}`` and ``{{call.<field>}}`` are filled as it is said.
            A call while a value a line reads is unset is refused, before anything is said.
        effect: ``ends_conversation`` ends the call once the tool has succeeded.
        interruption: What the caller's speech does to a running call. ``cancel`` drops
            it, ``continue`` lets it finish. ``provider_default`` is the framework's way:
            Pipecat cancels. LiveKit never cancels a running tool: when the reply that
            called it is cut off, its result reaches the model later.
        announce_wait: Run the tool only once the announce line has played. By default it
            runs at once, while the line plays.
        show_result: False gives the model ``{"ok": true}`` in place of a result that
            succeeded, for data the model must not read out. A failure is always shown.
        result_instructions: How the model should use the result, added to the
            description.
        hidden: Result fields the model never reads, like a line code speaks from the
            result. ``assign`` rows and spoken lines still read them. A failure is always
            shown whole.
        ends_as: Reads the tool's result and returns the end reason it gives the call, like
            ``transferred``, when its effect ends the conversation and it succeeded. None
            leaves the shared ``agent_ended``. The reason is told to the call's hooks.
        replies: The lines code speaks for the tool's own results, in place of a model
            reply, wherever it runs: on the main agent, in a task step, as a step's closing
            tool. Each reply leaves out ``tool``. A place's own replies are tried first.
    """

    name: Annotated[str, StringConstraints(pattern=f"^{TOOL_NAME.pattern}$")]
    description: Annotated[str, AfterValidator(_described)]
    parameters: Annotated[SkipValidation[dict[str, Any]], AfterValidator(check_parameters)]
    output: Annotated[type[BaseModel], PlainValidator(_output_model)]
    executor: Annotated[ToolExecutor, PlainValidator(_executes)]
    locked: SkipValidation[Mapping[str, object]] = field(default_factory=dict)
    announce: Annotated[SkipValidation[str | Sequence[str]], AfterValidator(_announce_lines)] = ()
    effect: ToolEffect = "returns_data"
    interruption: ToolInterruption = "provider_default"
    announce_wait: bool = False
    show_result: bool = True
    result_instructions: str = ""
    hidden: tuple[str, ...] = ()
    ends_as: SkipValidation[Callable[[object], str] | None] = None
    replies: Annotated[tuple[ToolReply, ...], Field(max_length=16)] = ()

    @model_validator(mode="after")
    def _locked_placeholders(self) -> Self:
        """Refuse a locked value that holds a placeholder that is not a value.

        Raises:
            ValueError: Naming the tool and the locked argument.
        """
        for name, source in self.locked.items():
            if not isinstance(source, str):
                continue
            if old := old_call_fact(source):
                raise ValueError(f"tool {self.name}: locked argument {name!r}: {old}")
            if _stray_braces(source):
                raise ValueError(
                    f"tool {self.name}: locked argument {name!r} holds a placeholder that is "
                    "not {{name}} or {{name.field}}"
                )
        return self

    @model_validator(mode="after")
    def _own_replies(self) -> Self:
        """Name this tool on its own replies, and refuse one that cannot play.

        Raises:
            ValueError: Naming the tool and what to fix.
        """
        if not self.replies:
            return self
        if self.effect == "ends_conversation":
            raise ValueError(
                f"tool {self.name} ends the call, so the call is over before a reply could "
                "play. Say a goodbye with its announce and announce_wait=True"
            )
        if stray := [reply.tool for reply in self.replies if reply.tool not in (None, self.name)]:
            raise ValueError(
                f"tool {self.name} has a reply for {stray[0]!r}. Leave tool out of a tool's "
                "own replies"
            )
        named = tuple(reply.model_copy(update={"tool": self.name}) for reply in self.replies)
        where = f"tool {self.name}: a reply"
        for reply in named:
            for path in result_paths(reply.say):
                self.check_result_path(path, where)
            self.check_match(reply.when, where)
        object.__setattr__(self, "replies", named)
        return self

    @model_validator(mode="after")
    def _hidden_fields(self) -> Self:
        """Refuse a hidden field the output model does not have.

        Raises:
            ValueError: Naming the tool and the field.
        """
        fields = {info.alias or name for name, info in self.output.model_fields.items()}
        if self.output.model_config.get("extra") == "allow":
            return self
        for name in self.hidden:
            if name not in fields:
                raise ValueError(
                    f"tool {self.name} hides {name!r}, which its output "
                    f"{self.output.__name__} does not have. It has {', '.join(sorted(fields))}"
                )
        return self

    @property
    def unlocked_required(self) -> list[str]:
        """The required arguments with no locked value: what only the model could send."""
        return [name for name in self.parameters.get("required", []) if name not in self.locked]

    @cached_property
    def model_description(self) -> str:
        """The description the model reads, with the result instructions after it."""
        if not self.result_instructions.strip():
            return self.description
        return f"{self.description}\n\nUse the result like this: {self.result_instructions.strip()}"

    def spoken(self, state: CallState | None) -> Self:
        """Return the tool with its announce lines filled, as a call says them.

        Args:
            state: The call state, for the ``{{name}}`` values.

        Returns:
            The tool, with filled lines. The same tool when no line holds a placeholder.

        Raises:
            ToolRefused: If a value a line reads is not set.
        """
        if not any("{{" in line for line in self.announce):
            return self
        filled = copy.copy(self)
        lines = (fill_text(line, state).strip() for line in self.announce)
        # A plain copy on purpose: the lines hold the caller's own words, which may hold
        # braces, and a re-validated copy would rebuild the cached argument validator.
        object.__setattr__(filled, "announce", tuple(line for line in lines if line))
        return filled

    def checked_output(self, result: object) -> dict[str, Any]:
        """Return a result as the output model reads it, as JSON data.

        Args:
            result: An instance of the output model, a mapping, or JSON text.

        Raises:
            ToolError: Naming the first field that does not fit. The model reads it.
        """
        if isinstance(result, self.output):
            return result.model_dump(mode="json")
        try:
            if isinstance(result, str | bytes):
                checked = self.output.model_validate_json(result)
            else:
                checked = self.output.model_validate(result)
        except ValidationError as error:
            first = error.errors()[0]
            where = ".".join(str(part) for part in first["loc"]) or "result"
            logger.warning(
                "tool %s returned a result that does not fit %s at %s",
                self.name,
                self.output.__name__,
                where,
            )
            raise ToolError(
                f"{self.name} returned a result that does not fit {self.output.__name__}: "
                f"{where}: {first['msg']}. Do not claim success.",
                event_text="the result does not fit the output model",
            ) from None
        return checked.model_dump(mode="json")

    def precheck(self, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Return the executor's precheck outcome as the output model reads it, or None.

        Args:
            arguments: The arguments, locked values filled.

        Raises:
            ToolError: If the arguments or the outcome do not fit.
        """
        if not isinstance(self.executor, Prechecking):
            return None
        outcome = self.executor.precheck(arguments)
        return None if outcome is None else self.checked_output(outcome)

    @cached_property
    def _output_schema(self) -> dict[str, Any]:
        """The output model's JSON Schema, built once."""
        return self.output.model_json_schema()

    def result_form(self, path: Sequence[str]) -> str | None:
        """Return how a value of the result is spoken: its ``format``, ``enum``, or None."""
        return form_at(self._output_schema, path)

    def input_form(self, path: Sequence[str]) -> str | None:
        """Return how an argument is spoken: its ``format``, ``enum``, or None."""
        return form_at(self.parameters, path)

    @cached_property
    def open_output(self) -> bool:
        """True when the output model takes fields it does not declare, so paths go unchecked."""
        return self.output.model_config.get("extra") == "allow"

    def check_result_path(self, path: Sequence[str], where: str) -> object:
        """Return the annotation of a path into the output model, or None for an open one.

        Args:
            path: Field names, like ``("appointment", "date")``. Empty is the whole result.
            where: What reads it, for the error.

        Raises:
            ValueError: If the output model has no such path.
        """
        if not path:
            return self.output
        if self.open_output:
            return None
        names = [info.alias or name for name, info in self.output.model_fields.items()]
        try:
            return path_annotation(self.output, parse_path(".".join(path), names))
        except ValuePathError as error:
            raise ValueError(
                f"{where} reads result.{'.'.join(path)}, which {self.output.__name__}, the "
                f"output of {self.name}, does not have: {error}"
            ) from None

    def check_match(self, fields: Mapping[str, Sequence[str | int | bool]], where: str) -> None:
        """Refuse a match on a field the output model lacks, or on a value it never holds.

        Args:
            fields: Result field to the values a rule matches, like a reply's ``when``.
            where: What matches, for the error.

        Raises:
            ValueError: Naming the field and the values it can hold.
        """
        for name, values in fields.items():
            annotation = self.check_result_path((name,), where)
            allowed = None if annotation is None else match_words(annotation)
            if allowed is None:
                continue
            if wrong := [value for value in values if success_word(value) not in allowed]:
                raise ValueError(
                    f"{where} matches result.{name} on {wrong[0]!r}, which {self.output.__name__} "
                    f"never holds there. It holds {', '.join(sorted(allowed))}"
                )

    @cached_property
    def reads_transcript(self) -> bool:
        """True when a locked argument is a ``Transcript``."""
        return any(isinstance(source, Transcript) for source in self.locked.values())

    @cached_property
    def takes_context(self) -> bool:
        """True when the executor's ``run`` takes ``context``: an event tool's event and transcript."""
        return "context" in inspect.signature(self.executor.run).parameters

    @cached_property
    def shown_parameters(self) -> dict[str, Any]:
        """The parameters without the locked arguments: what the model's arguments are checked against."""
        return model_schema(self.parameters, self.locked.keys())

    @cached_property
    def model_parameters(self) -> dict[str, Any]:
        """The parameters as the model sees them: ``shown_parameters``, shaped by ``model_shape``."""
        return model_shape(self.shown_parameters)

    @cached_property
    def _named_arguments(self) -> frozenset[str] | None:
        """The only arguments the model may give, or None when the schema takes others too.

        None too when the executor names stray arguments itself, in its own words.
        """
        schema = self.shown_parameters
        additional = schema.get("additionalProperties")
        if (
            getattr(self.executor, "checks_extra_arguments", False)
            or "properties" not in schema
            or additional is True
            or isinstance(additional, dict)
        ):
            return None
        return frozenset(schema["properties"])

    @cached_property
    def _validator(self) -> Validator:
        """The validator for the model's arguments."""
        return Draft202012Validator(self.shown_parameters)

    def check_arguments(self, arguments: dict[str, Any]) -> None:
        """Check the model's own arguments against the schema it was shown.

        A required argument left out, or sent as blank text, is missing: the model asks the
        caller for it, rather than reading a format error.

        Raises:
            ArgumentsMissing: Naming the required arguments left out or blank.
            ToolError: Naming the arguments the schema does not have, or the first one that
                does not fit.
        """
        if self._named_arguments is not None and (
            stray := sorted(arguments.keys() - self._named_arguments)
        ):
            many = len(stray) > 1
            raise ToolError(
                f"arguments: {', '.join(stray)} {'are not arguments' if many else 'is not an argument'}"
                f" of {self.name}. Call {self.name} again without {'them' if many else 'it'}."
            )
        if missing := [
            name
            for name in self.model_parameters.get("required", ())
            if name not in arguments
            or (isinstance(value := arguments[name], str) and not value.strip())
        ]:
            raise ArgumentsMissing(self.name, missing)
        if error := best_match(self._validator.iter_errors(arguments)):
            where = ".".join(str(part) for part in error.absolute_path) or "arguments"
            raise ToolError(f"{where}: {error.message}. Call {self.name} again with a fix.")

    def filled(
        self,
        own: dict[str, Any],
        state: CallState | None,
        context: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Return the model's arguments with the locked values filled, each as its type.

        Args:
            own: The model's own arguments, already checked.
            state: The call state.
            context: What only an event tool gets, for a ``Transcript``.

        Returns:
            The arguments the tool runs with.

        Raises:
            ToolError: If a locked value is not set, cannot be sent as its parameter's type,
                or the filled arguments do not fit the whole schema.
        """
        if not self.locked:
            return own
        locked = {}
        for name, source in self.locked.items():
            value = locked_value(source, state, context)
            try:
                locked[name] = coerce_locked(value, _parameter_type(self.parameters, name))
            except ToolError as error:
                raise ToolError(f"arguments.{name}: {error.message}") from None
        filled = own | locked
        if error := best_match(self._full_validator.iter_errors(filled)):
            where = ".".join(("arguments", *(str(part) for part in error.absolute_path)))
            raise ToolError(f"{where}: {error.message}")
        return filled

    @cached_property
    def _full_validator(self) -> Validator:
        """The validator for the arguments once the locked values are filled."""
        return Draft202012Validator(self.parameters)

    def check_locked(self, state: type[BaseModel]) -> None:
        """Check that each locked path names a plain value its parameter can take.

        Args:
            state: The State model. ``call`` is added to it, so ``{{call.<field>}}`` is checked.

        Raises:
            ValueError: Naming the locked argument and what is wrong.
        """
        state = call_model(state)
        names = [info.alias or name for name, info in state.model_fields.items()]
        for name, source in self.locked.items():
            if not isinstance(source, str):
                continue
            for match in PLACEHOLDER.finditer(source):
                where = f"tool {self.name}: locked argument {source!r}"
                try:
                    annotation = path_annotation(state, parse_path(match[1], names))
                except ValuePathError as error:
                    raise ValueError(f"{where}: {error}") from None
                expected = _parameter_type(self.parameters, name)
                whole = match[0] == source
                if not is_single_value(annotation):
                    if not whole:
                        raise ValueError(
                            f"{where} names a shape or a list inside a longer text. Lock it "
                            "whole, as the only text, to an object, array or untyped parameter"
                        )
                    wanted = "array" if is_list(annotation) else "object"
                    declared = self.parameters.get("properties", {}).get(name, {}).get("type")
                    if isinstance(declared, str) and declared != wanted:
                        raise ValueError(
                            f"{where} names a {'list' if wanted == 'array' else 'shape'}, "
                            f"so parameter {name!r} must be an {wanted}, not {declared}"
                        )
                    continue
                if whole and expected is not None and expected not in sendable_as(annotation):
                    raise ValueError(
                        f"{where} cannot be sent as the {expected} parameter {name!r}: its "
                        f"value is sent only as {', '.join(sorted(sendable_as(annotation)))}"
                    )


_TEXT_BASES = frozenset({"Id", "Phone", "Date", "Time", "EmailStr"})
"""The State value kinds held as text, which plain text from a result may fill."""


def _state_annotation(state: type[BaseModel], variable: str, where: str) -> object:
    """Return the annotation of one State value, or of a dotted path into its shape.

    Raises:
        ValueError: If the State has no such value, or the shape no such field.
    """
    fields = {info.alias or name: info for name, info in state.model_fields.items()}
    root = variable.split(".")[0]
    if root not in fields:
        raise ValueError(f"{where}: the State has no value {root!r}. It holds {', '.join(fields)}")
    if root == variable:
        return fields[variable].annotation
    try:
        return path_annotation(state, parse_path(variable, list(fields)))
    except ValuePathError as error:
        raise ValueError(f"{where}: {error}") from None


def _unfit(source: object, target: object, *, append: bool) -> str | None:
    """Return why a result value cannot be saved into a State value, or None when it fits.

    A whole shape fits only a State value of the same model class. Plain text fits any
    value held as text, like an id or a date, which the State checks when it is saved. A
    list is checked when it is saved, against the State's own types.
    """
    if append:
        if not is_list(target):
            return "that State value is not a list, so nothing can be appended to it"
        return None
    shape = _model_class(source)
    if shape is not None:
        if _model_class(target) is not shape:
            return f"that State value is not typed as {shape.__name__}"
        return None
    if is_list(source) or is_list(target):
        return None if is_list(source) and is_list(target) else "one of the two is a list"
    given, kept = value_base(source), value_base(target)
    if given == kept == "Literal":
        extra = set(literal_choices(source)) - set(literal_choices(target))
        return f"the State value does not take {sorted(map(str, extra))}" if extra else None
    if (
        kept == "str"
        or (given == "str" and kept in _TEXT_BASES)
        or (given, kept) == ("int", "float")
    ):
        return None
    if given != kept:
        return f"the result holds {given} and the State value holds {kept}"
    return None


def saved_values(rows: Sequence[Assign], result: object, state: CallState) -> dict[str, object]:
    """Return the values ``assign`` rows read from a result, ready for ``CallState.save``.

    An ``optional`` row with no value in the result is left out. An ``append`` row adds
    the value onto the list the State holds.
    """
    values: dict[str, object] = {}
    for row in rows:
        value = read_path(result, row.path)
        if value is None and row.optional:
            continue
        if row.append:
            current = list(values.get(row.variable, state.plain(row.variable)) or [])  # ty: ignore[invalid-argument-type]
            value = [*current, value] if value is not None else current
        values[row.variable] = value
    return values


class ToolRegistry:
    """The tools of an agent: their schemas for the model, and one way to call them.

    Adapters put ``schemas()`` into the model request and send every tool call to
    ``call``. They give the model ``for_model(name, result)`` and map ``ToolError`` to
    their framework's error.
    """

    def __init__(
        self,
        tools: Iterable[ToolSpec],
        *,
        state: type[BaseModel] | None = None,
        assign: Mapping[str, Sequence[Assign]] | None = None,
        replies: Sequence[ToolReply] = (),
    ) -> None:
        """Index the tools by name, and check what their results do.

        Args:
            tools: The tools.
            state: The State model. With it, each locked path is checked now: it names a
                plain value its parameter can take.
            assign: Tool name to the values its result saves into the State, on the main
                agent. A task step saves only what its own ``assign`` names.
            replies: The lines code speaks for a tool's result on the main agent, in
                place of a model reply. Each names its tool. The first that matches and
                fills is spoken. The tools' own ``replies`` follow them. A task step
                speaks its own ``replies``, then the tool's.

        Raises:
            ValueError: If two tools have the same name, a locked path does not fit, or an
                ``assign`` row or a reply names a tool, a result field or a State value
                that does not fit.
        """
        listed = list(tools)
        self._tools: dict[str, ToolSpec] = {tool.name: tool for tool in listed}
        self._sinks: list[ToolEventSink] = []
        if len(self._tools) != len(listed):  # the error names the first name seen twice
            names = [tool.name for tool in listed]
            twice = next(name for index, name in enumerate(names) if name in names[:index])
            raise ValueError(f"two tools are named {twice!r}. Give each a unique name")
        self.state = state
        self.assign: Mapping[str, tuple[Assign, ...]] = MappingProxyType(
            {name: tuple(rows) for name, rows in (assign or {}).items() if rows}
        )
        self.replies: tuple[ToolReply, ...] = (
            *replies,
            *(reply for tool in listed for reply in tool.replies),
        )
        if state is not None:
            for tool in self._tools.values():
                tool.check_locked(state)
        self._check_rows()

    def _check_rows(self) -> None:
        """Refuse an ``assign`` row or a reply that does not fit its tool or the State.

        Raises:
            ValueError: Naming the tool and what to fix.
        """
        for name, rows in self.assign.items():
            tool = self._owner(name, "assign")
            if self.state is None:
                raise ValueError(
                    f"assign: {name} saves to the State. Build the ToolRegistry with "
                    "state=YourState"
                )
            for row in rows:
                where = f"assign: {name} saves {row.variable}"
                if row.source == "input":
                    raise ValueError(
                        f"{where} from its input, but a tool's own save reads its result. "
                        "Save an input in the assign of the step that runs the tool"
                    )
                source = tool.check_result_path(row.path, where)
                target = _state_annotation(self.state, row.variable, where)
                if source is not None and (unfit := _unfit(source, target, append=row.append)):
                    raise ValueError(f"{where}, but {unfit}")
        for reply in self.replies:
            if reply.tool is None:  # only a tool's own replies leave it out
                raise ValueError(
                    "replies: a ToolReply on the ToolRegistry names its tool. Set tool=, or "
                    "put the reply on the tool's own replies"
                )
            tool = self._owner(reply.tool, "replies")
            if tool.effect == "ends_conversation":
                raise ValueError(
                    f"replies: {tool.name} ends the call, so the call is over before a line "
                    "could play. Say a goodbye with its announce and announce_wait=True"
                )
            where = f"replies: a line for {tool.name}"
            for path in result_paths(reply.say):
                tool.check_result_path(path, where)
            tool.check_match(reply.when, where)
            for variable in reply.say_variables:
                if self.state is None:
                    raise ValueError(
                        f"{where} reads the State value {variable!r}. Build the ToolRegistry "
                        "with state=YourState"
                    )
                _state_annotation(self.state, variable, where)

    def _owner(self, name: str, where: str) -> ToolSpec:
        """Return the tool a row names.

        Raises:
            ValueError: If the registry has no such tool.
        """
        if name not in self._tools:
            raise ValueError(
                f"{where}: there is no tool {name!r}. The tools are: {', '.join(self._tools)}"
            )
        return self._tools[name]

    def _copy(self, tools: Iterable[ToolSpec]) -> "ToolRegistry":
        """Return a registry of these tools with this one's State, rows and sinks.

        The rows of a tool it leaves out are dropped. Everything kept was checked already.
        """
        copied = ToolRegistry(tools)
        copied.state = self.state
        copied.assign = MappingProxyType(
            {name: rows for name, rows in self.assign.items() if name in copied}
        )
        copied.replies = tuple(reply for reply in self.replies if reply.tool in copied)
        copied._sinks = self._sinks
        return copied

    @property
    def names(self) -> tuple[str, ...]:
        """Every tool's name, in order."""
        return tuple(self._tools)

    def __contains__(self, name: object) -> bool:
        """Return True when the registry has a tool of this name."""
        return name in self._tools

    def keeps_running(self, name: str) -> bool:
        """Return True when the caller's speech must not cancel a running call of the tool.

        Args:
            name: The tool's name. A name the registry does not have is cancelled.
        """
        return name in self._tools and self._tools[name].interruption == "continue"

    def only(self, names: Iterable[str]) -> "ToolRegistry":
        """Return a registry of just these tools, like the ones one task step may call.

        Args:
            names: The tools to keep. Any iterable, read once.

        Raises:
            ValueError: If a name is not a tool of this registry.
        """
        wanted = list(names)
        if stray := [name for name in wanted if name not in self._tools]:
            raise ValueError(
                f"there is no tool {stray[0]!r}. The tools are: {', '.join(self._tools)}"
            )
        return self._copy(self._tools[name] for name in wanted)

    def with_tools(self, *tools: ToolSpec) -> "ToolRegistry":
        """Return a registry of these tools and the given ones, with the same State and sinks.

        Raises:
            ValueError: If a name is taken.
        """
        return self._copy([*self._tools.values(), *tools])

    def listen(self, sink: ToolEventSink) -> None:
        """Tell ``sink`` each call's stage. A registry from ``only`` tells the same sinks.

        A sink already listening is not added again.
        """
        if sink not in self._sinks:
            self._sinks.append(sink)

    def _emit(self, stage: str, event: _ToolEvent, hooks: ToolHooks | None = None) -> None:
        """Tell every sink, and the hooks. A sink that fails is logged and never breaks the call."""
        data = event.model_dump(exclude_none=True)
        for sink in self._sinks:
            try:
                sink(stage, data)
            except Exception:
                logger.exception("tool event sink failed on %s", stage)
        if hooks is not None:
            hooks.emit(stage, data)

    def get(self, name: str) -> ToolSpec:
        """Return one tool.

        Raises:
            ToolError: If there is no such tool.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"there is no tool {name!r}. The tools are: {', '.join(self._tools)}")
        return tool

    def ends_call(self, name: str, result: object) -> bool:
        """Return True when a call of the tool that gave ``result`` ends the call.

        Only a tool whose effect ends the conversation, and only when it succeeded: a
        result with an ``error`` field or ``ok: false`` leaves the call going.
        """
        tool = self._tools.get(name)
        return tool is not None and tool.effect == "ends_conversation" and tool_succeeded(result)

    def for_model(self, name: str, result: object) -> object:
        """Return what the model reads of a result: ``{"ok": true}`` for a hidden one.

        A tool with ``show_result=False`` hides a result that succeeded, and one with
        ``hidden`` fields leaves those out of it. A failure or a refusal is always shown
        whole, so the model never claims a success.
        """
        tool = self._tools.get(name)
        if tool is None or not tool_succeeded(result):
            return result
        if not tool.show_result:
            return dict(HIDDEN_RESULT)
        if tool.hidden and isinstance(result, Mapping):
            return {key: value for key, value in result.items() if key not in tool.hidden}
        return result

    def use_result(
        self,
        name: str,
        result: object,
        state: CallState | None,
        lines: SpokenLines,
        *,
        hooks: ToolHooks | None = None,
    ) -> str | None:
        """Save what a result's ``assign`` rows name, and pick the line code speaks for it.

        The main agent calls it after ``call``. A task step speaks only its own
        ``replies``: it calls ``save_result`` alone. Only a result that succeeded and does
        not end the call does anything.

        Args:
            name: The tool.
            result: What ``call`` returned.
            state: The call state.
            lines: The call's spoken lines.
            hooks: The call's hooks. They also hear a refused save.

        Returns:
            The line to speak in place of a model reply, or None.
        """
        if not self.save_result(name, result, state, hooks=hooks):
            return None
        tool = self._tools[name]
        line = lines.reply(tool, self.replies, result, state) if self.replies else None
        if line is not None:
            self.replied(name, hooks=hooks)
        return line

    def replied(self, name: str, *, hooks: ToolHooks | None = None) -> None:
        """Tell the sinks and hooks ``tool.replied``: code spoke a line for the tool's result.

        Args:
            name: The tool.
            hooks: The call's hooks.
        """
        self._emit("tool.replied", _ToolEvent(name=name, source="model"), hooks)

    def save_result(
        self,
        name: str,
        result: object,
        state: CallState | None,
        *,
        hooks: ToolHooks | None = None,
    ) -> bool:
        """Save what a result's ``assign`` rows name, at once.

        A save the State refuses is logged and told to the sinks as ``tool.save_refused``;
        the model still reads the result, since the tool ran. A tool's save never confirms
        a value.

        Args:
            name: The tool.
            result: What ``call`` returned.
            state: The call state.
            hooks: The call's hooks. They also hear a refused save.

        Returns:
            True when the result succeeded and does not end the call, saved or not.
        """
        tool = self._tools.get(name)
        if tool is None or not tool_succeeded(result) or self.ends_call(name, result):
            return False
        if state is not None and (rows := self.assign.get(name)):
            try:
                state.save(saved_values(rows, result, state), origin=StateOrigin("tool", name))
            except StateRefused as refused:
                logger.warning("tool %s: its result was not saved: %s", name, refused.message)
                event = _ToolEvent(name=name, source="model", error=refused.message)
                self._emit("tool.save_refused", event, hooks)
        return True

    def schemas(self) -> list[dict[str, Any]]:
        """Return every tool as an OpenAI function tool, for ``ModelRequest.tools``.

        Each call returns new dicts, so a hook that edits a request cannot change the next
        one. The description and the parameters inside are computed once per tool.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.model_description,
                    "parameters": tool.model_parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def call(
        self,
        name: str,
        arguments: str | Mapping[str, Any],
        state: CallState | None = None,
        *,
        context: Mapping[str, object] | None = None,
        started: ToolStarted | None = None,
        hooks: ToolHooks | None = None,
        source: ToolSource = "model",
    ) -> object:
        """Run one tool call from the model.

        The order is: read the arguments, the ``before_tool`` hooks, the checks and the
        locked values, the precheck, ``started``, the ``wrap_tool`` hooks around the tool,
        then the ``after_tool`` hooks. A precheck outcome is the result: the tool does not
        run and ``started`` is not called.

        Args:
            name: The tool the model called.
            arguments: The arguments, as the JSON text the model wrote or already parsed.
            state: The call state, or None for an agent without one.
            context: What only an event tool gets: its ``trigger_event`` and the
                ``transcript``. A ``Transcript`` locked value reads it. A call the model
                makes has none: its ``Transcript`` reads the hooks' transcript, when they
                are a ``TranscriptSource`` like the call's ``CallHooks``.
            started: Called with the tool, its announce lines filled, once the call is
                checked and just before it runs. A refused call never starts.
            hooks: The call's hooks. They also hear each stage.
            source: ``model``, or ``event`` for a tool on an event.

        Returns:
            The tool's result, as JSON data, as the ``after_tool`` hooks leave it. A
            result that succeeded is the output model's dump. A call that reads a value
            not set yet, or that a hook refuses, returns ``{"refused": message}`` and does
            not run. A call that leaves a required argument out or blank adds
            ``"missing": [names]`` to that.

        Raises:
            ToolError: If the call is invalid or fails, or its result does not fit the
                output model. Any other error of the tool or a hook is logged and becomes a
                ToolError that tells the model not to claim success.
        """
        try:
            tool, own = self._own(name, arguments)
        except ToolRefused as refused:
            refusal = _ToolEvent(name=name, source=source, error=refused.message)
            self._emit("tool.refused", refusal, hooks)
            return refused.result
        if context is None and tool.reads_transcript and isinstance(hooks, TranscriptSource):
            context = {"transcript": hooks.transcript()}
        run = ToolRun(name, own, source)
        try:
            if hooks is not None:
                run = await _before(hooks, run)
                tool.check_arguments(run.arguments)
            filled = tool.filled(run.arguments, state, context)
            if isinstance(tool.executor, ReadyChecking):
                tool.executor.check_ready(state)
            spoken = tool.spoken(state)
        except ToolRefused as refused:
            refusal = _ToolEvent(name=name, source=source, error=refused.message)
            self._emit("tool.refused", refusal, hooks)
            return refused.result
        self._emit("tool.started", _ToolEvent(name=name, source=source), hooks)

        async def execute(current: ToolRun) -> object:
            """Run the tool, and check what it returned against its output model.

            Its locked values are filled again when a hook changed the call. A result that
            reports an error is not checked: the model reads it as it is.
            """
            if current is not run:
                tool.check_arguments(current.arguments)
            given = filled if current is run else tool.filled(current.arguments, state, context)
            if (found := outcome if current is run else tool.precheck(given)) is not None:
                return found
            if tool.takes_context:
                result = await tool.executor.run(given, state, context=context)  # ty: ignore[unknown-argument]
            else:
                result = await tool.executor.run(given, state)
            if isinstance(result, BaseModel) or tool_succeeded(result):
                return tool.checked_output(result)
            return result

        begin = time.monotonic()
        outcome: dict[str, Any] | None = None
        try:
            outcome = tool.precheck(filled)
            if (
                outcome is None
                and started is not None
                and inspect.isawaitable(begun := started(spoken))
            ):
                await begun
            begin = time.monotonic()
            if hooks is None:
                result = await execute(run)
            else:
                result = await hooks.run_tool(run, execute)
                result = await _after(hooks, run, result)
        except ToolError as error:
            self._ended(run, begin, error.event_text, hooks)
            raise
        except asyncio.CancelledError:
            self._ended(run, begin, "cancelled", hooks)
            raise
        except Exception:
            logger.exception("tool %s failed", name)
            message = (
                f"{name} failed before completing. Do not claim success; retry only with "
                "corrected input."
            )
            self._ended(run, begin, message, hooks)
            raise ToolError(message) from None
        failure = None if tool_succeeded(result) else "the result reports a failure"
        self._ended(run, begin, failure, hooks)
        request = getattr(hooks, "end_requested", None)
        if failure is None and tool.ends_as and tool.effect == "ends_conversation" and request:
            request(tool.ends_as(result))
        return result

    def _ended(
        self, run: ToolRun, begin: float, error: str | None, hooks: ToolHooks | None = None
    ) -> None:
        """Tell the sinks a call ended, how long it took, and how."""
        ms = round((time.monotonic() - begin) * 1000)
        event = _ToolEvent(name=run.name, source=run.source, ms=ms, error=error)
        self._emit("tool.succeeded" if error is None else "tool.failed", event, hooks)

    def check(
        self,
        name: str,
        arguments: str | Mapping[str, Any],
        state: CallState | None = None,
        *,
        context: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        """Check a call's input as ``call`` would, without running the tool.

        Args:
            name: The tool the model called.
            arguments: The arguments, as JSON text or already parsed.
            state: The call state, for the locked values the check reads.
            context: What only an event tool gets: its ``trigger_event`` and the
                ``transcript``. A ``Transcript`` locked value reads it.

        Returns:
            The model's own arguments as the tool reads them, like a phone number in E.164.

        Raises:
            ToolError: If the input is refused, worded as ``call`` words it.
        """
        tool, own = self._own(name, arguments)
        if not isinstance(tool.executor, InputValidating):
            return own
        checked = tool.executor.validate(tool.filled(own, state, context))
        return {key: checked[key] for key in own if key in checked}

    def precheck(
        self, name: str, arguments: str | Mapping[str, Any], state: CallState | None = None
    ) -> dict[str, Any] | None:
        """Return a tool's precheck outcome for a call, without running the tool.

        A step runs it before it reads a closing tool's input back for a yes.

        Args:
            name: The tool the model called.
            arguments: The arguments, as JSON text or already parsed.
            state: The call state, for the locked values the precheck reads.

        Returns:
            The outcome as JSON data, or None when the tool may run.

        Raises:
            ToolError: If the input is refused, worded as ``call`` words it.
        """
        tool, own = self._own(name, arguments)
        return tool.precheck(tool.filled(own, state))

    def _own(
        self, name: str, arguments: str | Mapping[str, Any]
    ) -> tuple[ToolSpec, dict[str, Any]]:
        """Return the tool and the model's own arguments, parsed and checked against its schema."""
        tool = self.get(name)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                raise ToolError(
                    f"the arguments for {name} are not valid JSON. Call {name} again with a "
                    "JSON object"
                ) from None
        if not isinstance(arguments, Mapping):
            raise ToolError(
                f"the arguments for {name} must be a JSON object. Call {name} again with one"
            )
        own = {key: value for key, value in arguments.items() if key not in tool.locked}
        tool.check_arguments(own)
        return tool, own


async def _before(hooks: ToolHooks, run: ToolRun) -> ToolRun:
    """Run the ``before_tool`` hooks. A hook that fails stops the call with a ToolError.

    Raises:
        ToolRefused: If a hook refused the call.
        ToolError: If a hook failed.
    """
    try:
        return await hooks.run_before_tool(run)
    except ToolError:
        raise
    except Exception:
        logger.exception("a before_tool hook of %s failed", run.name)
        raise ToolError(f"{run.name} could not start. Do not claim success.") from None


async def _after(hooks: ToolHooks, run: ToolRun, result: object) -> object:
    """Run the ``after_tool`` hooks. A hook that fails is logged, and the tool's result kept.

    The tool already ran, so its call still counts as the result says.
    """
    try:
        return await hooks.run_after_tool(run, result)
    except Exception:
        logger.exception(
            "an after_tool hook of %s failed; the model reads the tool's result", run.name
        )
        return result
