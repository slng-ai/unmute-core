"""Typed call state for voice agents.

``CallState(State, confirm=...)`` runs on any Pydantic model: an author's own ``State``
class, or one ``StateSchema`` builds from declared variables and shapes written in the type
grammar (``Phone | None``, ``list[Contact]``). ``Phone``, ``Date``, ``Time``, ``Id`` and
``NameEmail`` are value types for such a model.

Every state also holds ``call``, a ``CallInfo``: what the call itself carries. Only the
library writes it. ``call_info(...)`` builds one for a phone call. Its ``agent_id`` is
required: pass it, or set ``UNMUTE_AGENT_ID``.
"""

from unmute_core.state._call_info import (
    AgentSettings,
    CallChannel,
    CallInfo,
    Direction,
    ReservedNameError,
    TrackingId,
    call_info,
    call_phone_region,
)
from unmute_core.state._call_state import (
    CallStartRule,
    CallState,
    HeldSave,
    NoValues,
    StateRefused,
)
from unmute_core.state._events import StateChange, StateEvent, StateListener, StateOrigin
from unmute_core.state._grammar import TypeExpr, TypeTextError, parse_type
from unmute_core.state._schema import Shape, ShapeField, StateSchema, StateSchemaError, Variable
from unmute_core.state._types import (
    BirthDate,
    Date,
    Id,
    NameEmail,
    Phone,
    Spelled,
    Time,
    region_of,
    use_phone_region,
)

__all__ = [
    "AgentSettings",
    "BirthDate",
    "CallChannel",
    "CallInfo",
    "CallStartRule",
    "CallState",
    "Date",
    "Direction",
    "HeldSave",
    "Id",
    "NameEmail",
    "NoValues",
    "Phone",
    "ReservedNameError",
    "Shape",
    "ShapeField",
    "Spelled",
    "StateChange",
    "StateEvent",
    "StateListener",
    "StateOrigin",
    "StateRefused",
    "StateSchema",
    "StateSchemaError",
    "Time",
    "TrackingId",
    "TypeExpr",
    "TypeTextError",
    "Variable",
    "call_info",
    "call_phone_region",
    "parse_type",
    "region_of",
    "use_phone_region",
]
