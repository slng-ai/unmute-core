"""The state of one call: typed values, their confirmation, and how they read as text.

Behaviour from unmute@d0c67f1 internal/generate/call_state.go (CallState: save_batch with
prefetch provenance and DEPENDENCIES, is_confirmed, withdraw_confirmation, plain, render).
CallState holds the model instead of subclassing it, so it works on any model class.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from typing import Any, Literal, Self, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass
from pydantic_core import to_json

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state._call_info import CALL, CallInfo, call_model, reserved_name_error
from unmute_core.state._events import StateChange, StateEvent, StateListener, StateOrigin

logger = logging.getLogger(__name__)

CallStartRule = Literal["required", "optional"]
"""Whether the call may start without a call-start value."""
CALL_START_RULES = frozenset(get_args(CallStartRule))
EMPTY_TEXT = "none recorded yet."
STORED_MAX = 64_000
HELD_DECISION_SECONDS = 4.0
"""How long the call's end waits on a step deciding a held save."""


@original_errors
@dataclass(frozen=True, config=VALUE)
class HeldSave:
    """A step's save refused before the caller spoke in it, kept for the call's end.

    The caller gave the answer before the step began, and the agent told them it was
    taken. A caller who hangs up then would lose it. It holds no serialized form,
    only what to call.

    Attributes:
        save: Saves the values: the step's decided ones, or given None, the held ones.
            It returns False when the values were refused.
        caller_spoke_since: True once the caller has said anything since the hold.
        decide: Asks the step, with what the caller said since, for the values to save,
            or None to save nothing. Without it, a held save the caller spoke after is
            dropped.
    """

    save: Callable[[Mapping[str, Any] | None], object]
    caller_spoke_since: Callable[[], bool]
    decide: Callable[[], Awaitable[Mapping[str, Any] | None]] | None = None


class StateRefused(Exception):  # noqa: N818 - Unmute's emitted code already uses this name
    """A value that does not fit its type, refused before anything is saved.

    Attributes:
        message: What was wrong, worded for the model, like ``count: Input should be ...``.
        name: The value it is about, like ``count``, or None when it names none.
        reason: What was wrong after the value's name: ``Input should be ...`` for
            ``count``, ``0.day: ...`` for ``bookings.0.day``. The whole message when it names
            no value.
    """

    def __init__(self, message: str, *, name: str | None = None, reason: str | None = None) -> None:
        """Keep the message the model will be shown, and the value it is about.

        Args:
            message: What was wrong with the value.
            name: The value it is about.
            reason: The message after the value's name. The whole message when None.
        """
        super().__init__(message)
        self.message = message
        self.name = name
        self.reason = message if reason is None else reason

    @classmethod
    def of(cls, name: str, reason: str) -> "StateRefused":
        """Refuse one value: the message reads ``name: reason``."""
        return cls(f"{name}: {reason}", name=name, reason=reason)

    @classmethod
    def from_error(cls, error: ValidationError) -> "StateRefused":
        """Word the first problem Pydantic found as ``path: message``."""
        first = error.errors()[0]
        path = [str(part) for part in first["loc"]]
        rest = path[1:]
        reason = f"{'.'.join(rest)}: {first['msg']}" if rest else first["msg"]
        name = path[0] if path else None
        return cls(f"{'.'.join(path)}: {first['msg']}", name=name, reason=reason)


class _StateRules(BaseModel):
    """The ``confirm``, ``dependencies`` and ``call_start`` rules, checked against the model.

    Validated with the context ``{"model": <class name>, "names": <value names>}``. A rule
    that breaks raises its own ``ValueError``, which ``CallState`` re-raises as itself.
    """

    model_config = ConfigDict(frozen=True)

    confirm: dict[str, str] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    call_start: dict[str, str] = {}

    @staticmethod
    def _unknown(rule: str, named: Collection[str], info: ValidationInfo) -> None:
        """Refuse a name the model does not have, and the library's own ``call``."""
        context = info.context or {}
        if CALL in named:
            raise reserved_name_error(rule)
        if unknown := sorted(set(named) - context["names"]):
            raise ValueError(f"{rule} names {unknown[0]!r}, which {context['model']} does not have")

    @field_validator("confirm")
    @classmethod
    def _confirm_names(cls, value: dict[str, str], info: ValidationInfo) -> dict[str, str]:
        """Refuse a confirmed value the model does not have."""
        cls._unknown("confirm", value, info)
        return value

    @field_validator("dependencies")
    @classmethod
    def _dependency_names(
        cls, value: dict[str, tuple[str, ...]], info: ValidationInfo
    ) -> dict[str, tuple[str, ...]]:
        """Refuse a prefetched value, or one it reads, that the model does not have."""
        reads = {read for reads in value.values() for read in reads}
        cls._unknown("dependencies", value.keys() | reads, info)
        return value

    @field_validator("call_start")
    @classmethod
    def _call_start_rules(cls, value: dict[str, str], info: ValidationInfo) -> dict[str, str]:
        """Refuse a call-start value the model does not have, and a rule that is no rule."""
        cls._unknown("call_start", value, info)
        # a plain str field, not the CallStartRule Literal: the refusal words the fix
        if bad := [rule for rule in value.values() if rule not in CALL_START_RULES]:
            raise ValueError(f"call_start rule {bad[0]!r} must be 'required' or 'optional'")
        return value

    @model_validator(mode="after")
    def _call_start_is_not_confirmed(self) -> Self:
        """Refuse a value the call starts with that also waits for the caller."""
        if both := sorted(self.call_start.keys() & self.confirm.keys()):
            raise ValueError(
                f"{both[0]!r} is in both call_start and confirm: a value the call starts with "
                "is never asked of the caller. Drop one"
            )
        return self


def _is_empty(value: object) -> bool:
    """Return True for a value with nothing in it. False and 0 are values."""
    return value is None or value == ""


def _origin_of(step: str | None, inputs: Sequence[str] | None) -> StateOrigin:
    """Name what saved a batch when the caller did not: the step, a prefetch, or the call."""
    if step is not None:
        return StateOrigin("step", step)
    return StateOrigin("prefetch") if inputs is not None else StateOrigin("call_start")


class NoValues(BaseModel):
    """The State of an agent that keeps no values of its own."""


def _stored_size(value: object) -> int:
    """Return how many characters a value holds: text as it is, anything else as compact JSON."""
    return len(value) if isinstance(value, str) else len(to_json(value).decode())


_EXACT_FLOAT = 2**53
"""Below this, a whole float and its int are the same number."""


def _whole_numbers(value: Any) -> Any:
    """Return ``value`` with each whole float written as an int, so 3.0 reads as 3.

    Only a float that holds the int exactly, below 2**53, is written as one.
    """
    if isinstance(value, float) and value.is_integer() and abs(value) < _EXACT_FLOAT:
        return int(value)
    if isinstance(value, list):
        return [_whole_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _whole_numbers(item) for key, item in value.items()}
    return value


class CallState[M: BaseModel]:
    """The typed values of one call, which of them the caller has confirmed, and their text.

    The model is the author's own ``State`` class, or one built at run time with
    ``create_model``. Every field needs a default, because every value starts empty. The
    library adds one value to it, ``call``: what the call carries (see ``CallInfo``).

    Attributes:
        values: The model instance that holds the typed values, ``call`` among them. Read
            it, never write it: ``save`` is the one way in.
        confirm: Value name to the step that must hear the caller agree to it.
        suppliers: Value name to the task or group the main agent starts to save it. The
            ``CallTasks`` of the agent that holds the call fills it (``CallTasks.activate``).
            A tool that reads an unset value tells the model to run that task first.
        dependencies: Prefetched value to the values it was looked up from. It is confirmed
            only while all of those are filled and confirmed.
        call_start: Value name to ``required`` or ``optional``: the values the call is
            dispatched with. ``save_call_start`` fills them. One that was not given reads as
            empty text, never "none recorded yet."
    """

    def __init__(
        self,
        model: type[M] = NoValues,  # ty: ignore[invalid-parameter-default]
        *,
        confirm: Mapping[str, str] | None = None,
        dependencies: Mapping[str, Sequence[str]] | None = None,
        call_start: Mapping[str, CallStartRule] | None = None,
        call: CallInfo | None = None,
    ) -> None:
        """Start a call with every value at its default and nothing confirmed.

        Args:
            model: The Pydantic model class of the state. None of its own by default.
            confirm: Value name to the step that confirms it.
            dependencies: Prefetched value name to the value names it was looked up from.
            call_start: Value name to ``required`` or ``optional``, for the values the
                call is dispatched with.
            call: What the call carries. A web call's, with a fresh ``session_id`` and
                ``UNMUTE_AGENT_ID`` as its ``agent_id``, when None.

        Raises:
            TypeError: If ``model`` is not a Pydantic model class, or a field has no default.
            ReservedNameError: If the model, ``confirm``, ``dependencies`` or ``call_start``
                names ``call``.
            ValueError: If ``confirm``, ``dependencies`` or ``call_start`` names a value the
                model does not have, a ``call_start`` rule is neither required nor optional,
                or a value is in both ``call_start`` and ``confirm``.
        """
        # inspects the model class and raises TypeError, which a field constraint cannot
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError(
                f"CallState needs a Pydantic model class, got {model!r}. Pass your State "
                "class, or StateSchema(...).model()"
            )
        self._model = call_model(model)
        self._fields = {
            info.alias or name: name
            for name, info in self._model.model_fields.items()
            if name != CALL
        }
        self._slots = self._fields | {CALL: CALL}
        if required := [name for name, info in model.model_fields.items() if info.is_required()]:
            raise TypeError(
                f"{model.__name__}.{required[0]} has no default. Every value starts the call "
                "empty, so give it one, like `= None`"
            )
        names = frozenset(self._fields)
        try:
            rules = _StateRules.model_validate(
                {
                    "confirm": confirm or {},
                    "dependencies": dependencies or {},
                    "call_start": call_start or {},
                },
                context={"model": model.__name__, "names": names},
            )
        except ValidationError as error:
            own = error.errors()[0].get("ctx", {}).get("error")
            if isinstance(own, ValueError):
                raise own from None
            raise
        self.confirm = dict(rules.confirm)
        self.dependencies = dict(rules.dependencies)
        self.call_start: dict[str, CallStartRule] = dict(rules.call_start)  # ty: ignore[invalid-assignment]
        self._names = tuple(self._fields)
        self._name_set = names
        self.suppliers: dict[str, str] = {}
        self.values: M = self._model.model_validate({CALL: call} if call is not None else {})
        self._unconfirmed = set(self.confirm)
        self._provenance: dict[str, tuple[str, ...]] = {}
        self._held: dict[str, HeldSave] = {}
        self._listeners: list[StateListener] = []
        self._origins: dict[str, StateOrigin] = {}

    @property
    def model(self) -> type[M]:
        """The State model the values follow, with the library's ``call`` added."""
        return self._model

    @property
    def call(self) -> CallInfo:
        """What the call carries: its numbers, direction and ids. Read-only."""
        return getattr(self.values, CALL)

    def description(self, name: str) -> str | None:
        """Return the description of a value in the State model, or None without one.

        Args:
            name: The value's name.
        """
        field = self._fields.get(name)
        return self._model.model_fields[field].description if field is not None else None

    @property
    def names(self) -> tuple[str, ...]:
        """Every value name of the author's model, in its order. ``call`` is not one."""
        return self._names

    @property
    def name_set(self) -> frozenset[str]:
        """The same names as a set, for a membership test."""
        return self._name_set

    def set_suppliers(self, suppliers: Mapping[str, str]) -> None:
        """Name the task or group that supplies each value, replacing the earlier names.

        Args:
            suppliers: Value name to the task or group the main agent starts to save it.
        """
        self.suppliers = dict(suppliers)

    def clear_suppliers(self) -> None:
        """Forget every supplier: the agent that holds the call has no tasks."""
        self.suppliers = {}

    def save(
        self,
        values: Mapping[str, object],
        *,
        step: str | None = None,
        inputs: Sequence[str] | None = None,
        origin: StateOrigin | None = None,
    ) -> None:
        """Check a batch of values as a whole, then save all of it, or nothing.

        A confirmed value stays confirmed only when its confirming step saves it with
        something in it. Any other save of it waits for the caller again.

        A value saved with ``inputs`` remembers them. When one of those inputs later changes,
        the value goes back to its default, and so does anything looked up from it.

        Args:
            values: Value name to what to save.
            step: The step saving them.
            inputs: The value names a prefetch read to produce them.
            origin: What is saving them, as listeners are told. It defaults to the ``step``, or
                a ``prefetch`` when ``inputs`` is given, or else the call's start. Only an
                origin of kind ``call``, the library's, may save ``call``.

        Raises:
            StateRefused: If a value is unknown, does not fit its type, or is too long, or
                the batch saves ``call`` without the library's origin.
        """
        if not values:
            return
        pending, _ = self._checked(values, origin)
        changed = {name for name, value in pending.items() if self._value(name) != value}
        while (
            derived := {
                name for name, reads in self._provenance.items() if changed.intersection(reads)
            }
            - changed
        ):
            changed |= derived
        invalidated = changed - pending.keys()
        provenance = {
            name: reads for name, reads in self._provenance.items() if name not in invalidated
        }
        unconfirmed = set(self._unconfirmed)
        for name, value in pending.items():
            if inputs is None:
                provenance.pop(name, None)
            else:
                provenance[name] = tuple(inputs)
            if name in self.confirm and (self.confirm[name] != step or _is_empty(value)):
                unconfirmed.add(name)
            else:
                unconfirmed.discard(name)

        def current(name: str) -> object:
            """Read a value as it will stand once this batch lands."""
            if name in invalidated:
                return self._initial(name)
            return pending[name] if name in pending else self._value(name)

        unconfirmed = self._settle(unconfirmed, current)
        for name in invalidated:
            setattr(self.values, self._slots[name], self._initial(name))
        for name, value in pending.items():
            setattr(self.values, self._slots[name], value)
        flipped = unconfirmed ^ self._unconfirmed
        self._unconfirmed = unconfirmed
        self._provenance = provenance
        by = origin or _origin_of(step, inputs)
        changed_here = [name for name in pending if name in changed]
        self._origins |= dict.fromkeys(changed_here, by)
        self._origins |= dict.fromkeys(invalidated, StateOrigin("reset", by.name))
        self._tell(by, changed_here, sorted(invalidated), sorted(flipped))

    def check(self, values: Mapping[str, object]) -> dict[str, Any]:
        """Check values by the rules a save applies, and save none of them.

        Use it to test what a tool or a step would save before acting on it, or to find
        which values of a batch fit, so the rest can be saved on their own.

        Args:
            values: Value name to what would be saved.

        Returns:
            Each value as the state would hold it, as JSON data, like ``plain`` reads it.

        Raises:
            StateRefused: For the first value that a save would refuse, with the same text.
        """
        if not values:
            return {}
        _, plain = self._checked(values, None)
        return plain

    def _checked(
        self, values: Mapping[str, object], origin: StateOrigin | None
    ) -> tuple[dict[str, object], dict[str, Any]]:
        """Validate a batch as a save would.

        Returns:
            Each value typed, as the State model holds it, and each as JSON data.

        Raises:
            StateRefused: If a value is unknown, does not fit its type, or is too long, or the
                batch holds ``call`` without the library's origin.
        """
        if CALL in values and (origin is None or origin.kind != "call"):
            raise StateRefused.of(
                CALL,
                "the library sets it from the call itself. Copy a call value into one of your "
                "own to change it",
            )
        if unknown := [name for name in values if name not in self._slots]:
            raise StateRefused.of(
                unknown[0], f"not a value of this state. It holds {', '.join(self._fields)}"
            )
        try:
            checked = self._model.model_validate({CALL: self.call, **values})
        except ValidationError as error:
            raise StateRefused.from_error(error) from None
        dumped = checked.model_dump(
            mode="json", by_alias=True, include={self._slots[name] for name in values}
        )
        for name in values:
            # measures the value's own text and raises StateRefused, the error the model is shown
            if (size := _stored_size(dumped[name])) > STORED_MAX:
                raise StateRefused.of(
                    name,
                    f"too long to keep: {size} characters, and a value holds at most {STORED_MAX}",
                )
        typed = {name: getattr(checked, self._slots[name]) for name in values}
        return typed, {name: dumped[name] for name in values}

    def _settle(self, unconfirmed: set[str], current: Callable[[str], object]) -> set[str]:
        """Confirm each prefetched value only while all its sources are filled and confirmed.

        Args:
            unconfirmed: The values waiting for the caller, before dependencies are applied.
            current: Reads a value as it will stand after the save.

        Returns:
            The values waiting for the caller, dependencies applied.
        """
        while True:
            settled = {
                name
                for name, reads in self.dependencies.items()
                if not _is_empty(current(name))
                and all(read not in unconfirmed and not _is_empty(current(read)) for read in reads)
            }
            after = (unconfirmed - self.dependencies.keys()) | (self.dependencies.keys() - settled)
            if after == unconfirmed:
                return after
            unconfirmed = after

    def _value(self, name: str) -> object:
        """Return the typed value saved under a name."""
        return getattr(self.values, self._slots[name])

    def _initial(self, name: str) -> object:
        """Return a fresh copy of the value a name starts the call with."""
        return self._model.model_fields[self._slots[name]].get_default(call_default_factory=True)

    def is_confirmed(self, name: str) -> bool:
        """Return True when the value is filled and its confirmation stands."""
        return name not in self._unconfirmed and not _is_empty(self.plain(name))

    def is_unconfirmed(self, name: str) -> bool:
        """Return True while the value waits for the caller to agree to it."""
        return name in self._unconfirmed

    def withdraw(self, step: str) -> None:
        """Withdraw every confirmation this step gave, because the step runs again.

        Values looked up from a withdrawn value wait for the caller too.
        """
        before = set(self._unconfirmed)
        self._unconfirmed |= {name for name, owner in self.confirm.items() if owner == step}
        while (
            derived := {
                name
                for name, reads in self.dependencies.items()
                if self._unconfirmed.intersection(reads)
            }
            - self._unconfirmed
        ):
            self._unconfirmed |= derived
        self._tell(StateOrigin("withdraw", step), [], [], sorted(self._unconfirmed - before))

    def listen(self, listener: StateListener, *, replay: bool = False) -> Callable[[], None]:
        """Tell a listener about every later save that changed something.

        The listener gets one ``StateEvent`` per save: the values that changed, with what
        they are now, whether the caller has confirmed them, and what changed each. A save
        that changed nothing sends nothing. A listener that raises is logged and removed. It
        never fails the save.

        Args:
            listener: Called with each event.
            replay: True to be told first, as one event of origin ``start``, the values
                already set, when something was set before the listener came.

        Returns:
            A function that stops the listening. Calling it twice does nothing.
        """
        self._listeners.append(listener)
        if replay and (started := self._started()) is not None:
            self._deliver(listener, started)

        def stop() -> None:
            """Remove the listener."""
            if listener in self._listeners:
                self._listeners.remove(listener)

        return stop

    def _started(self) -> StateEvent | None:
        """Return the values set so far as one ``start`` event, or None when none are."""
        names = [name for name in self._fields if name in self._origins]
        changes = tuple(self._change(name, self._origins[name]) for name in names)
        return StateEvent(StateOrigin("start"), changes) if changes else None

    def _change(self, name: str, by: StateOrigin) -> StateChange:
        """Describe a value as it stands: its JSON, and whether the caller confirmed it."""
        waits = name in self.confirm or name in self.dependencies
        return StateChange(name, self.plain(name), self.is_confirmed(name) if waits else None, by)

    def _tell(
        self,
        origin: StateOrigin,
        changed: Sequence[str],
        reset: Sequence[str],
        flipped: Sequence[str],
    ) -> None:
        """Tell each listener what a save or a withdrawal changed. Free when none listens.

        Args:
            origin: What made the change.
            changed: The values the save set to something new.
            reset: The values that went back to their default.
            flipped: The values whose confirmation changed. One that holds nothing is left out.
        """
        if not self._listeners:
            return
        by = {name: StateOrigin("reset", origin.name) for name in reset}
        by |= dict.fromkeys(changed, origin)
        by |= {
            name: origin for name in flipped if name not in by and not _is_empty(self.plain(name))
        }
        if not by:
            return
        event = StateEvent(origin, tuple(self._change(name, who) for name, who in by.items()))
        for listener in tuple(self._listeners):
            self._deliver(listener, event)

    def _deliver(self, listener: StateListener, event: StateEvent) -> None:
        """Call one listener. One that raises is logged and removed."""
        try:
            listener(event)
        except Exception:
            logger.warning("a state listener failed and was removed", exc_info=True)
            if listener in self._listeners:
                self._listeners.remove(listener)

    def hold(self, owner: str, held: HeldSave) -> None:
        """Keep a step's refused save for the call's end, replacing one it held before."""
        self._held[owner] = held

    def release(self, owner: str) -> None:
        """Drop a step's held save: it finished, or decides again."""
        self._held.pop(owner, None)

    async def settle_held(self, *, timeout_s: float = HELD_DECISION_SECONDS) -> list[str]:
        """Settle every held save, once the caller has hung up. Never raises.

        Call it when the call ends, before telemetry closes. A held save the caller said
        nothing after is saved as it was: they heard it taken and did not correct it. One
        they spoke after is saved only as its step decides from what they said: "thanks,
        goodbye" keeps it, "no, that is wrong" does not. A decision that fails or runs out
        of time saves nothing. A value saved here is never confirmed: no step was left to
        confirm it.

        Args:
            timeout_s: How long each step may take to decide.

        Returns:
            The owners whose saves were made.
        """
        held, self._held = self._held, {}
        settled = []
        for owner, item in held.items():
            try:
                if not item.caller_spoke_since():
                    made = item.save(None)
                elif item.decide is None:
                    logger.info("held save of %s dropped: the caller spoke after it", owner)
                    continue
                elif (decided := await asyncio.wait_for(item.decide(), timeout_s)) is None:
                    logger.info("held save of %s dropped: the caller did not keep it", owner)
                    continue
                else:
                    made = item.save(decided)
            except Exception:
                logger.exception("held save of %s could not be made", owner)
                continue
            if made is not False:
                settled.append(owner)
        return settled

    def plain(self, path: str) -> Any:
        """Return a value as JSON data. A dotted path reads into a shape.

        Args:
            path: A value name, or a dotted path like ``customer.status``.

        Returns:
            JSON data, so a date reads ``"2026-10-02"``. None past an absent link.

        Raises:
            KeyError: If the path starts with a name the state does not have.
        """
        return self._read(path)

    def _read(self, path: str, dumped: Mapping[str, Any] | None = None) -> Any:
        """Read ``plain`` from a dump of the values, or dump just the one value it needs."""
        root, *parts = path.split(".")
        if root not in self._slots:
            raise KeyError(f"no state value named {root!r}. It holds {', '.join(self._slots)}")
        if dumped is None:
            dumped = self.values.model_dump(mode="json", by_alias=True, include={self._slots[root]})
        value = dumped[root]
        for part in parts:
            value = value.get(part) if isinstance(value, dict) else None
        return value

    def render(self, path: str, site: str = "") -> str:
        """Return a value as a prompt reads it.

        Text as it is, anything else as compact JSON. An empty value, or one still waiting
        for the caller, reads as "none recorded yet." The step that confirms a value still
        sees it, so it can read it back to the caller. An empty call-start value, and an
        empty ``call`` field, read as empty text: nobody will record them later.

        Args:
            path: A value name, or a dotted path into a shape.
            site: The prompt reading it, like ``task:verify_customer``.

        Returns:
            The text. A model call with its own limit, like the router's, cuts it.
        """
        return self._render(path, site)

    def _render(self, path: str, site: str, dumped: Mapping[str, Any] | None = None) -> str:
        """Write ``render`` from a dump of the values, or dump just the one it reads.

        A value waiting for the caller reads empty, except in the prompt of the step that
        confirms it. A looked-up value no step confirms reads empty everywhere until its
        sources are confirmed.
        """
        root = path.split(".")[0]
        if root in self._unconfirmed and site != f"task:{self.confirm.get(root)}":
            return EMPTY_TEXT
        value = self._read(path, dumped)
        if _is_empty(value):
            return "" if root in self.call_start or root == CALL else EMPTY_TEXT
        return value if isinstance(value, str) else to_json(_whole_numbers(value)).decode()

    def snapshot(self, site: str = "") -> dict[str, str]:
        """Return every value as ``render`` gives it, for the router's template variables.

        ``call`` is left out: a ``{{call.<field>}}`` placeholder is rendered on its own.
        """
        dumped = self.dump()
        return {name: self._render(name, site, dumped) for name in self._fields}

    def dump(self) -> dict[str, Any]:
        """Return the author's values as JSON data, without ``call``, as telemetry keeps them."""
        return self.values.model_dump(mode="json", by_alias=True, exclude={CALL})
