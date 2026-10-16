"""Values a call is dispatched with, saved into the state before the call starts.

Ported from unmute@2c78a5a internal/generate/templates/livekit_v1/session.py.tmpl:77-152
and internal/generate/call_state.go:566-585: the dispatch's ``call_start`` object, with
``UNMUTE_CALL_START`` under it for a local run. Unknown names are dropped, a missing
required name or a value that does not fit refuses the whole batch. A ``call`` object sets
what the call carries, like a simulated caller on a web call.
"""

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import Field, ValidationError

from unmute_core.settings import EnvSettings
from unmute_core.state import CallInfo, CallState, StateOrigin, StateRefused
from unmute_core.state._call_info import CALL

LINE_FIELDS = frozenset(
    {"channel", "direction", "calling_number", "called_number", "carrier_call_id"}
)
"""What a phone line itself reports. On a phone call a dispatch only fills the empty ones."""

logger = logging.getLogger(__name__)


class CallStartError(ValueError):
    """The call cannot start: a required value is missing, or a value does not fit."""


class CallStartSettings(EnvSettings):
    """Call-start values for a local run, when no dispatch carries them.

    Attributes:
        values: A JSON object, like ``{"customer_id": "c_42"}``.
    """

    values: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="UNMUTE_CALL_START",
        description=(
            "Call-start values as a JSON object, for a local run. A 'call' object sets what "
            'the call carries, like {"call": {"direction": "inbound", '
            '"calling_number": "+34612345678"}}.'
        ),
    )


def save_call_start(
    state: CallState, dispatched: Mapping[str, object] | None, *, env: bool = True
) -> dict[str, object]:
    """Save the values the call was dispatched with, or nothing.

    ``CallState(call_start=...)`` names the values a dispatch may set, and the ones the call
    cannot start without. Without it, a dispatch may set any State value. A name the State
    does not take is dropped. A ``None`` value is left out: the value keeps its default.

    A ``call`` object sets fields of ``state.call``. On a web call every field it gives
    applies, so a web session can simulate an inbound or outbound phone call. On a phone
    call the line wins: the dispatch fills only the line's fields it left empty. Its
    ``agent_id`` and ``session_id`` always apply.

    Args:
        state: The call's state.
        dispatched: The dispatch's ``call_start`` object. None when there is none.
        env: Also read ``UNMUTE_CALL_START``, a JSON object that the dispatch wins over.
            False reads only ``dispatched``.

    Returns:
        The values saved.

    Raises:
        CallStartError: If a required value is missing, or a value does not fit its type.
        SettingsError: If ``UNMUTE_CALL_START`` is not a JSON object.
    """
    values = (CallStartSettings().values if env else {}) | dict(dispatched or {})
    if (call := values.pop(CALL, None)) is not None:
        _save_call(state, call)
    required = [name for name, rule in state.call_start.items() if rule == "required"]
    if missing := [name for name in required if values.get(name) in (None, "")]:
        raise CallStartError(f"Missing call_start fields: {', '.join(missing)}")
    allowed = state.call_start.keys() if state.call_start else set(state.names)
    if dropped := sorted(name for name in values if name not in allowed):
        logger.warning("call_start: %s is no call-start value; dropped", ", ".join(dropped))
    kept = {name: value for name, value in values.items() if name in allowed and value is not None}
    try:
        state.save(kept, origin=StateOrigin("call_start"))
    except StateRefused as refused:
        raise CallStartError(f"call_start: {refused.message}") from None
    return kept


def _save_call(state: CallState, given: object) -> None:
    """Save the fields a dispatch gives for ``state.call``, the line's own winning on a phone.

    Raises:
        CallStartError: If ``call`` is not an object, or a field is unknown or does not fit.
    """
    if not isinstance(given, Mapping):  # the env and a dispatch are untyped JSON
        raise CallStartError(f"call_start: call must be an object, not {given!r}")
    current = state.call
    phone = current.channel == "phone"
    kept = {
        name: value
        for name, value in given.items()
        if not (phone and name in LINE_FIELDS and getattr(current, name, None) is not None)
    }
    try:
        call = CallInfo.model_validate(current.model_dump(exclude={"remote_number"}) | kept)
    except ValidationError as error:
        raise CallStartError(
            f"call_start: call: {StateRefused.from_error(error).message}"
        ) from None
    state.save({CALL: call}, origin=StateOrigin("call", "call_start"))
