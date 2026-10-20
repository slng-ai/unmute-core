"""Errors a model call raises before any model receives the request."""


class RequestRefusedError(ValueError):
    """A model call refused a request before it sent it, so no model received it.

    A request option, a body field or a template variable the call cannot send is one.
    Sending the same request again cannot help, but the next call of a chain has its own
    settings and may take it. So ``FallbackModelCall`` moves to the next call at once, with
    no retry.
    """
