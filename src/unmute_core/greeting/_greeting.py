"""How a call opens: a fixed line, one the model words, or silence until the caller speaks.

Ported from Unmute's ``conversation.greeting`` (unmute@868ae23 internal/spec/package.go:827,
internal/ir/validate.go:618-624, templates/livekit_v1/agents.py.tmpl:301-324). A fixed line
may name state values. Only values prefetch fills before the greeting can be named: any
other value is still empty when the call opens.
"""

import logging
from collections.abc import Iterable
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, model_validator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.prefetch import Prefetch
from unmute_core.state import CallState
from unmute_core.state._call_info import CALL
from unmute_core.state._call_state import EMPTY_TEXT
from unmute_core.state._paths import PLACEHOLDER, ValuePathError, parse_path

logger = logging.getLogger(__name__)

GREETING_INSTRUCTION = "Greet the caller and offer to help."
"""What the model is told when it words the greeting, word for word as Unmute's."""
GREETING_HOLD_SECONDS = 20.0
"""The longest time a greeting holds the caller's audio."""


def _check_placeholders(text: str) -> str:
    """Return the line once every ``{{`` and ``}}`` in it belongs to a placeholder.

    Raises:
        ValueError: If a brace pair is not ``{{name}}`` or ``{{name.field}}``.
    """
    left = PLACEHOLDER.sub("", text)
    if "{{" in left or "}}" in left:
        raise ValueError("greeting text holds a placeholder that is not {{name}} or {{name.field}}")
    return text


@original_errors
@dataclass(frozen=True, config=VALUE)
class Greeting:
    """How the call opens.

    Attributes:
        speaks_first: ``agent`` opens the call; ``user`` waits for the caller.
        text: The line the agent says, which may name values as ``{{name}}``. None has
            the model word the greeting.
        instruction: What the model is told when it words the greeting.
        interruptible: False keeps the caller from cutting the greeting off, and holds the
            caller's audio until it has played.
    """

    speaks_first: Literal["agent", "user"] = "agent"
    text: Annotated[str, Field(pattern=r"\S"), AfterValidator(_check_placeholders)] | None = None
    instruction: str = GREETING_INSTRUCTION
    interruptible: bool = True

    @model_validator(mode="after")
    def _text_needs_the_agent_first(self) -> Self:
        """Refuse a line when the caller speaks first.

        Raises:
            ValueError: Naming the field to fix.
        """
        if self.text is not None and self.speaks_first == "user":
            raise ValueError("greeting text requires speaks_first: agent")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """The value paths the line names, in order."""
        return tuple(match[1] for match in PLACEHOLDER.finditer(self.text or ""))

    def check(self, prefetch: Prefetch | None, call_start: Iterable[str] = ()) -> None:
        """Refuse a line that names a value nothing fills before it. ``call`` is filled already.

        Args:
            prefetch: The prefetch that fills values before the greeting, or None.
            call_start: The values the call starts with: the names of
                ``CallState(call_start=...)``, or that mapping. ``save_call_start`` saves
                them before the greeting. An optional one that is missing leaves the line
                to the model.

        Raises:
            ValueError: Naming the value and the fix.
        """
        fills = (prefetch.fills if prefetch is not None else frozenset()) | set(call_start)
        for name in self.names:
            try:
                root = parse_path(name).root
            except ValuePathError as error:
                raise ValueError(f"greeting text: {error}") from None
            if root not in fills and root != CALL:
                raise ValueError(
                    f"greeting text names {{{{{name}}}}}, which neither a prefetch entry nor "
                    "call_start fills: it is still empty when the call opens. Fill it in "
                    "prefetch, name it in call_start, or leave it out"
                )

    def line(self, state: CallState | None) -> str | None:
        """Return the line to say, its values filled, or None to have the model word it.

        A line whose value is still empty, because its prefetch failed, or still waits for
        the caller to confirm it, is not said: the model words the greeting instead, never
        "Hello none recorded yet".

        Args:
            state: The call's state, to read the values from, or None when it has none.

        Returns:
            The line to say, or None.
        """
        if self.speaks_first == "user" or self.text is None:
            return None
        if not self.names:
            return self.text
        if state is None:
            logger.warning("greeting names values but the call has no state; the model words it")
            return None
        texts = {name: state.render(_dotted(name)) for name in self.names}
        if empty := [name for name, text in texts.items() if text in (EMPTY_TEXT, "")]:
            logger.warning(
                "greeting: %s empty or unconfirmed; the model words it", ", ".join(empty)
            )
            return None
        return PLACEHOLDER.sub(lambda match: texts[match[1]], self.text)


def _dotted(name: str) -> str:
    """Return a value path in its dotted form: ``a__b`` reads as ``a.b``."""
    return parse_path(name).text
