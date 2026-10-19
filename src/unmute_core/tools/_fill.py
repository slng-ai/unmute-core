"""Locked arguments and spoken lines: filling ``{{name}}`` values from the call state.

Each function reads a state, so they live apart from ``ToolSpec``. A ``{{name}}`` or
``{{name.field}}`` is a State value. ``{{call.<field>}}`` is what the call carries.
"""

import json
import re
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import Field
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state import CallState
from unmute_core.state._call_info import CALL
from unmute_core.state._paths import PLACEHOLDER, ValuePathError, parse_path
from unmute_core.tools._errors import ToolError, ToolRefused

OLD_CALL_FACT = re.compile(r"\{\{\s*@([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
"""A ``{{@call_id}}``: the call facts' old spelling, refused with the new one."""
_RENAMED = MappingProxyType(
    {
        "call_id": "{{call.carrier_call_id}} or {{call.session_id}}",
        "phone_number": "{{call.remote_number}}",
        "call_end_reason": "{{call.end_reason}}",
        "transcript": "a Transcript() locked argument",
        "first_user_message": "{{call.first_user_message}}",
    }
)


def old_call_fact(text: str) -> str | None:
    """Return why a ``{{@name}}`` in ``text`` is refused, naming what to write, or None."""
    if (match := OLD_CALL_FACT.search(text)) is None:
        return None
    new = _RENAMED.get(match[1], "{{call.<field>}}, a value of state.call")
    return f"{match[0]} is gone: call values live in the state now. Write {new}"


def _path(name: str, state: CallState | None) -> str:
    """Turn a placeholder name into a state path: ``customer__phone`` reads ``customer.phone``.

    Args:
        name: The placeholder's name.
        state: The call state, whose value names the path may use.

    Returns:
        The dotted path.

    Raises:
        ToolError: If the name is no path.
    """
    try:
        return parse_path(name, state.names if state is not None else ()).text
    except ValuePathError as error:
        raise ToolError(str(error)) from None


def _unset_hint(root: str, state: CallState) -> str:
    """Say what sets a value: its confirming task, else the task that saves it, else the caller."""
    task = state.confirm.get(root)
    if task is not None and state.is_unconfirmed(root):
        return f"Run {task} first: it confirms {root} with the caller"
    if (supplier := state.suppliers.get(root)) is not None:
        return f"Run {supplier} first: it saves {root}"
    if (description := state.description(root)) is not None:
        return f"Get it from the caller ({description})"
    return "Get it from the caller"


def require_set(paths: Iterable[str], state: CallState | None) -> None:
    """Refuse the call while a value it reads is empty or waits for the caller's yes.

    An optional call-start value never refuses the call: it reads as empty text.

    Args:
        paths: State paths, like ``customer_phone`` or ``customer.phone``.
        state: The call state.

    Raises:
        ToolRefused: Naming every value that is not set, and what sets it.
        ToolError: If the agent keeps no state.
    """
    paths = list(dict.fromkeys(paths))
    if not paths:
        return
    if state is None:
        raise ToolError(
            f"this tool reads {', '.join(paths)}, but this agent keeps no state. Pass a "
            "CallState, or lock the argument to a literal"
        )
    if absent := [p for p in paths if p.split(".")[0] == CALL and state.plain(p) in (None, "")]:
        raise ToolRefused(f"this tool reads {absent[0]}, which this call does not carry")
    unset = [
        path
        for path in paths
        if state.call_start.get(path.split(".")[0]) != "optional"
        and (not state.is_confirmed(path.split(".")[0]) or state.plain(path) in (None, ""))
    ]
    if unset:
        hints = dict.fromkeys(_unset_hint(path.split(".")[0], state) for path in unset)
        raise ToolRefused(
            f"{', '.join(unset)} not set yet. {'. '.join(hints)}, then call this tool again."
        )


def fill_text(text: str, state: CallState | None, encode: Callable[[str], str] = str) -> str:
    """Fill each ``{{name}}`` or ``{{name.field}}`` with the value's text, after ``require_set``.

    A value's own text is never filled again.

    Args:
        text: Text with placeholders, like ``/customers/{{customer_id}}``.
        state: The call state.
        encode: Applied to each value's text, like ``urllib.parse.quote``.

    Returns:
        The filled text.

    Raises:
        ToolError: If a value is not set.
    """
    require_set((_path(match[1], state) for match in PLACEHOLDER.finditer(text)), state)
    if state is None:
        return text
    return PLACEHOLDER.sub(lambda match: encode(state.render(_path(match[1], state))), text)


TRANSCRIPT_MAX_BYTES = 32 * 1024
"""The most a ``Transcript`` sends: the oldest messages go first when it is over."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class Transcript:
    """A locked argument that reads the call's transcript so far.

    It reads the transcript the adapter hands the tool: chat messages with ``role`` and
    ``content``.

    Attributes:
        max_messages: The newest messages to send, at least one.
        text: True sends one text, a ``role: content`` line per message. False sends the
            messages as a list.
        max_bytes: The most bytes to send, as the request body encodes them. Over it the
            oldest messages are dropped, so the newest always go. The default is
            ``TRANSCRIPT_MAX_BYTES``. None sets no cap of its own. A webhook request is
            still limited, whatever this says.
    """

    max_messages: Annotated[int, Field(ge=1)] = 200
    text: bool = False
    max_bytes: Annotated[int | None, Field(ge=1)] = TRANSCRIPT_MAX_BYTES

    def read(self, context: Mapping[str, object] | None) -> list[dict[str, Any]] | str:
        """Return the newest messages of the call, as a list or as text."""
        found = (context or {}).get("transcript") or []
        messages = (
            [m for m in found if isinstance(m, dict)] if isinstance(found, list | tuple) else []
        )
        newest = messages[-self.max_messages :]
        if not self.text:
            if self.max_bytes is None:
                return newest
            encoded = [json.dumps(m, default=str) for m in newest]
            return _within_budget(newest, encoded, 2, 2, self.max_bytes)
        lines = [f"{m.get('role', '')}: {m.get('content', '')}" for m in newest]
        if self.max_bytes is None:
            return "\n".join(lines)
        escaped = [json.dumps(line)[1:-1] for line in lines]
        return "\n".join(_within_budget(lines, escaped, 2, 2, self.max_bytes))


def _within_budget(
    items: list[Any], encoded: list[str], overhead: int, gap: int, limit: int
) -> list[Any]:
    """Drop the oldest items until their encoding fits ``limit`` bytes.

    Args:
        items: The items, oldest first.
        encoded: Each item as the text that is sent, JSON-escaped as a request body sends it.
        overhead: Bytes around the whole (the brackets of a JSON list).
        gap: Bytes between two items.
        limit: The most bytes the whole may take.

    Returns:
        The newest items that fit, at least the newest one.
    """
    sizes = [len(text.encode()) for text in encoded]
    total = overhead + sum(sizes) + gap * max(len(sizes) - 1, 0)
    start = 0
    while total > limit and start < len(items) - 1:
        total -= sizes[start] + gap
        start += 1
    return items[start:]


def locked_value(
    source: object, state: CallState | None, context: Mapping[str, object] | None = None
) -> object:
    """Return a locked argument's value.

    A whole ``"{{customer_phone}}"`` or ``"{{call.calling_number}}"`` is the typed value, so
    a number stays a number. Other text with placeholders is filled as text. Anything else
    is the value itself.

    A ``Transcript`` is the call's transcript so far.

    Args:
        source: The locked value as the tool declares it.
        state: The call state.
        context: What only an event tool gets: its ``trigger_event`` and the ``transcript``.

    Returns:
        The value to send.

    Raises:
        ToolError: If a value it reads is not set.
    """
    if isinstance(source, Transcript):
        return source.read(context)
    if not isinstance(source, str):
        return source
    if not PLACEHOLDER.search(source):
        return source
    if whole := PLACEHOLDER.fullmatch(source):
        path = _path(whole[1], state)
        require_set([path], state)
        assert state is not None
        value = state.plain(path)
        if value is None and state.call_start.get(path.split(".")[0]) == "optional":
            return ""
        return value
    return fill_text(source, state)
