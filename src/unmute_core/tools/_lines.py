"""What a tool's result does on a call: the values it saves, and the line code says for it.

``Assign`` saves one value of a result into the State. ``ToolReply`` is a line code says
when a tool returns a matching result, so the model is asked for no reply. A task step
lists both on its ``TaskSpec``. An agent without tasks lists them on its ``ToolRegistry``.

``SpokenLines`` fills those lines for one call. A ``{{result.<field>}}`` is said by its
JSON Schema ``format``, in the call's language, the way ``spoken`` says it.
"""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import field
from datetime import date, datetime
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Literal, get_args, get_origin
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state import CallState
from unmute_core.state._call_state import EMPTY_TEXT
from unmute_core.state._paths import unwrapped
from unmute_core.tools._spoken import spoken

if TYPE_CHECKING:
    from unmute_core.tools._spec import ToolSpec

IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
"""One name of a State value, or one field of a tool's result."""
RECEIPT_PLACEHOLDER = re.compile(
    rf"\{{\{{\s*((?:result|input)(?:\.{IDENTIFIER})+|{IDENTIFIER}(?:\.{IDENTIFIER})*)\s*\}}\}}"
)
"""A placeholder of a spoken line: ``{{result.<path>}}``, ``{{input.<path>}}``, or a State
value, whole or by a dotted path into its shape, like ``{{account.amount_words}}``."""

VariableName = Annotated[str, StringConstraints(pattern=rf"^{IDENTIFIER}$", max_length=64)]
"""The name of a value of the State."""

FieldName = Annotated[str, StringConstraints(pattern=rf"^{IDENTIFIER}$", max_length=200)]
"""One field of a tool's result."""

ToolName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
"""A tool name, as ``ToolSpec`` names it."""

MatchValues = Annotated[tuple[str | int | bool, ...], Field(min_length=1, max_length=64)]
"""The values that count for one field."""

SpokenLine = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""One line code speaks, stripped."""


def read_path(value: object, path: Sequence[str]) -> object:
    """Walk a result by field names. None past a missing link. An empty path is the result."""
    return reduce(
        lambda node, part: node.get(part) if isinstance(node, Mapping) else None, path, value
    )


def success_word(value: object) -> str:
    """How a value reads against a match: booleans as true and false."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def _matches(fields: Mapping[str, Sequence[object]], read: Callable[[str], object]) -> bool:
    """Return True when every field holds one of the values listed for it."""
    return all(
        success_word(read(name)) in {success_word(value) for value in values}
        for name, values in fields.items()
    )


def match_words(annotation: object) -> frozenset[str] | None:
    """Return the words a match on a field of this type can hold, or None when any can.

    A ``Literal`` holds its values, a ``bool`` true and false, an ``Enum`` its values.
    ``Annotated`` and ``| None`` are looked through.
    """
    annotation = unwrapped(annotation)
    if annotation is bool:
        return frozenset({"true", "false"})
    if get_origin(annotation) is Literal:
        return frozenset(success_word(value) for value in get_args(annotation))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return frozenset(success_word(member.value) for member in annotation)
    return None


def result_matches(fields: Mapping[str, Sequence[str | int | bool]], result: object) -> bool:
    """Return True when every field of the result holds one of the values listed for it."""
    return isinstance(result, Mapping) and _matches(fields, result.get)


def spoken_lines(lines: tuple[str, ...], source: Literal["result", "input"]) -> tuple[str, ...]:
    """Refuse a spoken line that holds a placeholder it cannot fill."""
    other = "input" if source == "result" else "result"
    for line in lines:
        left = RECEIPT_PLACEHOLDER.sub(
            lambda match: match[0] if match[1].startswith(f"{other}.") else "", line
        )
        if "{{" in left or "}}" in left:
            raise ValueError(
                f"a {'confirm' if source == 'input' else 'spoken'} line takes only "
                f"{{{{{source}.<field>}}}} and {{{{variable}}}} placeholders"
            )
    return lines


def say_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Check lines spoken from a tool's result."""
    return spoken_lines(lines, "result")


def say_variables(lines: Iterable[str]) -> set[str]:
    """Return the State values spoken lines read, leaving out ``{{result...}}`` and ``{{input...}}``.

    A value read by a dotted path, like ``account.amount_words``, is returned as that path.
    """
    return {
        match[1]
        for line in lines
        for match in RECEIPT_PLACEHOLDER.finditer(line)
        if not match[1].startswith(("result.", "input."))
    }


def result_paths(lines: Iterable[str]) -> set[tuple[str, ...]]:
    """Return the result paths spoken lines read: ``{{result.appointment.date}}`` reads one."""
    return {
        tuple(match[1].split(".")[1:])
        for line in lines
        for match in RECEIPT_PLACEHOLDER.finditer(line)
        if match[1].startswith("result.")
    }


class _Row(BaseModel):
    """Immutable, and refuses keys it does not know."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Assign(_Row):
    """One value a result saves: ``variable`` takes the result's ``path``.

    Attributes:
        variable: The State value it saves.
        path: Where the value sits in the result, like ``("order", "id")``. An empty path
            saves the whole result, into a State value typed as the tool's output model.
        append: Add one entry onto a list instead of replacing it (Unmute's ``+``).
        optional: A step may finish without it (Unmute's ``?``). Otherwise a served
            finish must carry it, unless it appends. A tool's own save skips it when the
            result holds no value there.
        source: ``result`` reads the path from what the tool returned. ``input`` reads it
            from the input the model gave the step's closing tool, saved when that tool
            succeeds: the booking the caller said yes to, when the result does not echo
            it. Only a task's ``assign`` takes ``input``.
    """

    variable: VariableName
    path: Annotated[tuple[FieldName, ...], Field(max_length=8)]
    append: bool = False
    optional: bool = False
    source: Literal["result", "input"] = "result"

    @property
    def argument(self) -> str:
        """The finish parameter: the field for a one-part path, otherwise the variable."""
        return self.path[0] if len(self.path) == 1 else self.variable


class ToolReply(_Row):
    """A line code speaks when a tool returns a matching result.

    The model reads the result, and is asked for no reply: the line is the reply. Use it
    for a line that must come out word for word, like the times on offer. A question the
    caller must say yes to before a tool runs is that tool's ``FinishRule.confirm``.

    A value a line reads from the result is said by its output model's JSON Schema
    ``format``, in the call's language: a ``date`` is "tomorrow, October 9".

    Attributes:
        tool: The tool. Left out on a tool's own ``replies``: it is that tool.
        when: Result field to the values that call for the line. Every field must hold
            one of its values. Empty matches every result. On a ``Literal``, ``bool`` or
            ``Enum`` field, each value must be one the field can hold.
        state: State value to the values that call for the line, read the same way.
        say: Lines spoken from the result, with ``{{result.<field>}}`` and
            ``{{variable}}`` placeholders. One is picked each time.
        assign: Values the matching result saves into the State, as a tool's own
            ``assign`` saves them. Only the first reply that matches saves.
    """

    tool: ToolName | None = None
    when: Annotated[dict[FieldName, MatchValues], Field(max_length=16)] = Field(
        default_factory=dict
    )
    state: Annotated[dict[VariableName, MatchValues], Field(max_length=16)] = Field(
        default_factory=dict
    )
    say: Annotated[tuple[SpokenLine, ...], Field(min_length=1), AfterValidator(say_lines)]
    assign: tuple[Assign, ...] = ()

    @property
    def say_variables(self) -> set[str]:
        """The State values the spoken lines and the ``state`` match read."""
        return say_variables(self.say) | self.state.keys()

    def matches(self, result: object, state: CallState | None) -> bool:
        """Return True when the result and the State hold what the reply calls for."""
        if not result_matches(self.when, result):
            return False
        if not self.state:
            return True
        return state is not None and _matches(self.state, state.plain)


def _word(value: object) -> str | None:
    """Return a value as a spoken line says it, or None when it has none."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).strip() or None


def render_line(line: str, words: Mapping[str, object]) -> str | None:
    """Fill a spoken line, or None when a value it reads is missing.

    Args:
        line: The line, with ``{{result.x}}``, ``{{input.x}}`` and ``{{variable}}``.
        words: Each placeholder's name, like ``result.due``, to its value as said.
    """
    missing = False

    def fill(match: re.Match[str]) -> str:
        """Return one placeholder's word, and note when it has none."""
        nonlocal missing
        word = _word(words.get(match[1]))
        missing = missing or word is None
        return word or ""

    filled = RECEIPT_PLACEHOLDER.sub(fill, line)
    return None if missing else filled


@original_errors
@dataclass(eq=False, config=VALUE)
class ReceiptLines:
    """Per call: which spoken line each site said last, so the next one differs."""

    _next: dict[str, int] = field(init=False, default_factory=dict)

    def pick(self, key: str, lines: Sequence[str], words: Mapping[str, object]) -> str | None:
        """Return the next line that fills, or None.

        Args:
            key: The site, like ``manage_booking:save_booking``.
            lines: The lines to speak, in turn.
            words: Each placeholder's name to its value as said.
        """
        start = self._next.get(key, 0)
        for offset in range(len(lines)):
            index = (start + offset) % len(lines)
            if (filled := render_line(lines[index], words)) is not None:
                self._next[key] = index + 1
                return filled
        return None


def zone_of(timezone: str | None) -> ZoneInfo | None:
    """Return the zone an IANA name stands for, or None for no name.

    Raises:
        ValueError: When the name is no IANA zone.
    """
    if not timezone:
        return None
    try:
        return ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"timezone {timezone!r} is not an IANA name, like Europe/Madrid") from None


class SpokenLines:
    """The lines code speaks on one call, in the call's language. Build one per call.

    Attributes:
        language: The call's language, a BCP-47 tag like ``es-MX``. Set it again when the
            caller switches.
        zone: Where the business is, or None.
    """

    def __init__(self, language: str = "en", timezone: str | None = None) -> None:
        """Start with no line said yet.

        Args:
            language: The call's language, a BCP-47 tag like ``es-MX``.
            timezone: Where the business is, an IANA name like ``Europe/Madrid``. A date
                is then named from today: "today, October 8". Without it a date is said in
                full, with its year.

        Raises:
            ValueError: When the timezone is unknown.
        """
        self.language = language
        self.zone = zone_of(timezone)
        self._receipts = ReceiptLines()

    def today(self) -> date | None:
        """Return today's date where the business is, or None without a timezone."""
        return datetime.now(self.zone).date() if self.zone else None

    def say(self, value: object, form: str | None) -> object:
        """Return one value as code says it, by its JSON Schema ``format``."""
        return spoken(value, form, self.language, self.today(), self.zone)

    def line(
        self,
        key: str,
        lines: Sequence[str],
        data: object,
        form: Callable[[tuple[str, ...]], str | None],
        values: Mapping[str, object],
    ) -> str | None:
        """Return the next of ``lines`` that fills, or None.

        Args:
            key: The site, which keeps its own turn of lines.
            lines: The lines, in turn.
            data: The tool's result for a ``{{result.x}}``, or its input for ``{{input.x}}``.
            form: The JSON Schema ``format`` of a path into the data.
            values: The State values a line may read, as text.
        """
        if not lines or not isinstance(data, Mapping):
            return None
        words: dict[str, object] = dict(values)
        for line in lines:
            for match in RECEIPT_PLACEHOLDER.finditer(line):
                name = match[1]
                if name.startswith(("result.", "input.")) and name not in words:
                    path = tuple(name.split(".")[1:])
                    words[name] = self.say(read_path(data, path), form(path))
        return self._receipts.pick(key, lines, words)

    def reply(
        self,
        tool: "ToolSpec",
        replies: Sequence[ToolReply],
        result: object,
        state: CallState | None,
        *,
        site: str = "",
    ) -> str | None:
        """Return the line of the first reply that matches the result and fills, or None.

        Args:
            tool: The tool that returned.
            replies: The replies to try, in order. Those of other tools are skipped.
            result: The tool's result.
            state: The call state, for ``{{variable}}`` and ``ToolReply.state``.
            site: Keeps these replies' turns apart from another owner's, like a step's.
        """
        values: dict[str, str] | None = None
        for index, reply in enumerate(replies):
            if reply.tool != tool.name or not reply.matches(result, state):
                continue
            values = state_words(state, site, reply.say) if values is None else values
            key = f"{site}:{tool.name}:{index}"
            if (line := self.line(key, reply.say, result, tool.result_form, values)) is not None:
                return line
        return None


def state_words(
    state: CallState | None, site: str = "", lines: Iterable[str] = ()
) -> dict[str, str]:
    """Return the State values a line may read, as text, leaving out the unset ones.

    Args:
        state: The call state.
        site: The prompt reading them, like ``task:verify``. A value waiting for the
            caller reads only where it is confirmed.
        lines: The lines to read for. Each dotted path they name, like
            ``{{account.amount_words}}``, is added as it renders.
    """
    if state is None:
        return {}
    words = {name: text for name, text in state.snapshot(site).items() if text != EMPTY_TEXT}
    for path in say_variables(lines):
        if "." in path and (text := state.render(path, site)) not in (EMPTY_TEXT, ""):
            words[path] = text
    return words
