"""The errors a tool call raises, apart from the modules that raise them."""

from collections.abc import Sequence
from typing import Any


class ToolError(Exception):
    """A tool call failed in a way the model should hear. Adapters map it to their framework.

    Attributes:
        message: What went wrong, worded for the model.
        event_text: A shorter line for the tool event sinks, when the message holds
            something they must not carry, like a body another service sent.
    """

    def __init__(self, message: str, *, event_text: str | None = None) -> None:
        """Keep the message the model will be shown.

        Args:
            message: What went wrong.
            event_text: What the event sinks read instead of the message.
        """
        super().__init__(message)
        self.message = message
        self.event_text = event_text or message


class ToolRefused(ToolError):  # noqa: N818 - the model reads {"refused": ...}, as Unmute emits
    """A tool call did not run, because a value it reads is not set yet.

    ``ToolRegistry.call`` gives the model ``{"refused": message}``, so it asks the caller
    for the value and calls again. Nothing is said or sent first.
    """

    @property
    def result(self) -> dict[str, Any]:
        """What the model reads in place of the tool's result."""
        return {"refused": self.message}


class ArgumentsMissing(ToolRefused):
    """The model left required arguments out, or sent them empty.

    Small models often call a tool before they have asked the caller, with ``""`` for
    what they do not know. The model reads ``{"refused": ..., "missing": [names]}``.

    Attributes:
        names: The missing arguments.
    """

    def __init__(self, tool: str, names: Sequence[str]) -> None:
        """Word the refusal for the model.

        Args:
            tool: The tool the model called.
            names: The missing arguments, in schema order.
        """
        many = len(names) > 1
        listed = " and ".join((", ".join(names[:-1]), names[-1])) if many else names[0]
        super().__init__(
            f"{listed} {'are' if many else 'is'} missing. Ask the caller for "
            f"{'them' if many else 'it'}, then call {tool} again."
        )
        self.names = tuple(names)

    @property
    def result(self) -> dict[str, Any]:
        """The refusal, and the names a reply rule reads."""
        return {**super().result, "missing": list(self.names)}
