"""What a call reads as it starts, saved into a ``CallState``: before the greeting, or beside it.

``Prefetch([Clock(...), CallValue(...), Lookup(...)], state=State).run(state)`` runs the
entries in order inside one budget of 2 seconds. A ``Lookup(background=True)`` waits for
``run_background``, which runs beside the greeting. A lookup may run a tool. A
``CallValue`` copies a ``state.call`` value, like the caller's number, into a value of your
own. Nothing here can fail a call.
"""

from unmute_core.prefetch._prefetch import (
    BACKGROUND_BUDGET_S,
    BUDGET_S,
    CALL_VALUE_SOURCES,
    CLOCK_FIELDS,
    CallValue,
    CallValueSource,
    Clock,
    ClockField,
    Lookup,
    Prefetch,
)

__all__ = [
    "BACKGROUND_BUDGET_S",
    "BUDGET_S",
    "CALL_VALUE_SOURCES",
    "CLOCK_FIELDS",
    "CallValue",
    "CallValueSource",
    "Clock",
    "ClockField",
    "Lookup",
    "Prefetch",
]
