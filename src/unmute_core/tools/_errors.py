"""The errors a tool call raises, apart from the modules that raise them."""


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
