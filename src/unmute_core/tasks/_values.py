"""A step's finish values: their model, their schema, success rules, and the one-batch save.

The result model follows Unmute's ``StepResult`` (unmute@4a738ad internal/generate/
call_state.go:690-948): one Pydantic model per step, built from the State's own field
types, so the model is asked for exactly what the save accepts. ``save_task_values``
saves them in one batch, into ``CallState``.
"""

import json
import logging
from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import Annotated, Any, Literal, TypedDict, get_args, get_origin

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
)
from pydantic.dataclasses import dataclass
from pydantic_core import to_jsonable_python

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.state import CallState, StateOrigin, StateRefused
from unmute_core.state._grammar import safe_field_name
from unmute_core.state._types import json_text
from unmute_core.tasks._spec import UNSERVED_FIELD, Assign, FinishRule, TaskSpec, read_path
from unmute_core.tasks._texts import UNSERVED_DESCRIPTION, UNSERVED_DESCRIPTION_RECORD
from unmute_core.tools._lines import result_matches
from unmute_core.tools._schema import model_shape

logger = logging.getLogger(__name__)


def _is_empty(value: object) -> bool:
    """Return True for None, an empty string, an empty list or an empty dict: no value."""
    return value is None or (isinstance(value, str | list | dict) and not value)


class TaskValueRefused(Exception):  # noqa: N818 - Unmute's emitted code says "refused"
    """A finish value that does not fit; nothing was saved.

    Attributes:
        message: What was wrong, worded for the model, like ``status: Input should be ...``.
    """

    def __init__(self, message: str) -> None:
        """Keep the message the model will be shown.

        Args:
            message: What was wrong.
        """
        super().__init__(message)
        self.message = message


@original_errors
@dataclass(frozen=True, config=VALUE)
class SaveOutcome:
    """How a step ended.

    Attributes:
        status: ``completed`` saved. ``unserved`` handed the caller back. ``failed``
            broke, or ended on a finish rule that ends ``failed`` and saved only its rows.
        saved: The values it saved.
        unserved_request: What it could not serve, in the caller's words.
        refused: Values a hand-back was given that did not fit, left as they were.
    """

    status: Literal["completed", "unserved", "failed"]
    saved: tuple[str, ...] = ()
    unserved_request: str = ""
    refused: tuple[str, ...] = ()


def _no_request(value: object) -> object:
    """Read None as the empty request: the model sends nothing for a step it served."""
    return "" if value is None else value


def _blank_is_absent(value: object) -> object:
    """Read an empty string as no value, for a field whose type has no empty value."""
    return None if value == "" else value


class StepResult(BaseModel):
    """What one step's finish carries. Each step builds its own subclass (``result_model``).

    Every field may be left out. An empty string reads as no value, unless the field's type
    takes one: the model sends empty for a field it has nothing for.
    """

    unserved_request: Annotated[str, BeforeValidator(_no_request)] = Field(
        "", description=UNSERVED_DESCRIPTION
    )


def _field_type(model: type[BaseModel], name: str) -> Any:
    """Return the annotation a State field validates against, constraints included."""
    info = model.model_fields[name]
    return Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation


def item_type(annotation: Any) -> Any | None:
    """Return the entry type of a list annotation, through Annotated and None, or None."""
    for candidate in (annotation, *get_args(annotation)):
        if get_origin(candidate) is list:
            return get_args(candidate)[0]
        if get_origin(candidate) is Annotated and (found := item_type(get_args(candidate)[0])):
            return found
    return None


def _accepts_blank(annotation: Any) -> bool:
    """Return True when an empty string is a value of the type, as for text.

    Not cached: a shape type is a class built per call, and a cache would keep it alive.
    It runs only when a plan is built.
    """
    try:
        TypeAdapter(annotation).validate_python("")
    except ValidationError:
        return False
    return True


def state_fields(model: type[BaseModel]) -> dict[str, str]:
    """Return each State value's public name (its alias, or its name) to its field name."""
    return {info.alias or name: name for name, info in model.model_fields.items()}


def _aliased(name: str, description: str | None) -> Any:
    """Return the field info of an optional value: its real name is the alias when needed."""
    return Field(
        None, description=description, alias=name if safe_field_name(name) != name else None
    )


def _result_field(
    state: type[BaseModel], fields: Mapping[str, str], row: Assign
) -> tuple[Any, Any]:
    """Return the model field of one assignment: its State type, optional, and its text."""
    annotation: Any = _field_type(state, fields[row.variable])
    if row.append and (item := item_type(annotation)) is not None:
        annotation = item
    optional: Any = annotation | None
    if not _accepts_blank(annotation):
        optional = Annotated[optional, BeforeValidator(_blank_is_absent)]
    return optional, _aliased(row.argument, state.model_fields[fields[row.variable]].description)


def result_model(task: TaskSpec, state: type[BaseModel]) -> type[StepResult]:
    """Build the finish model of one step from the State's field types.

    An appended value is one entry of the list. Every field may be None.

    Args:
        task: The task.
        state: The State model.

    Returns:
        The step's own ``StepResult`` subclass.
    """
    fields = state_fields(state)
    definitions: dict[str, Any] = {
        safe_field_name(row.argument): _result_field(state, fields, row) for row in task.assign
    }
    title = "".join(part.title() for part in task.name.split("_")) + "Result"
    return create_model(title, __base__=StepResult, **definitions)


def finish_parameters(
    model: type[StepResult], *, record: bool, derived: Collection[str] = ()
) -> dict[str, Any]:
    """Return the finish tool's parameters: one optional property per value, refs inlined.

    Args:
        model: The step's result model.
        record: The mode is ``record``, whose hand-back keeps confirmed values.
        derived: Parameters read from a parent value (``derived_arguments``), not offered.
    """
    parameters = _optional_parameters(model)
    left_out = {*derived, UNSERVED_FIELD}
    parameters["properties"] = {
        name: node for name, node in parameters["properties"].items() if name not in left_out
    } | {
        UNSERVED_FIELD: {
            "type": "string",
            "description": UNSERVED_DESCRIPTION_RECORD if record else UNSERVED_DESCRIPTION,
        }
    }
    return parameters


def _optional_parameters(model: type[BaseModel]) -> dict[str, Any]:
    """Return a tool's parameters from a model: one optional property per field, refs inlined."""
    schema = model.model_json_schema(by_alias=True)
    return model_shape(
        {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": [],
            "additionalProperties": False,
            "$defs": schema.get("$defs", {}),
        }
    )


def argument_parameters(names: Sequence[str], state: type[BaseModel]) -> dict[str, Any]:
    """Return a delegate tool's parameters: each State value it takes, typed and optional.

    Args:
        names: The State values, as the spec names them.
        state: The State model, whose field types and descriptions the model reads.
    """
    fields = state_fields(state)
    definitions: dict[str, Any] = {
        safe_field_name(name): (
            _field_type(state, fields[name]) | None,
            _aliased(name, state.model_fields[fields[name]].description),
        )
        for name in names
    }
    return _optional_parameters(create_model("DelegateArguments", **definitions))


def given_arguments(names: Sequence[str], arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Return every value a delegate takes from the model's arguments: None where it gave none.

    An empty string is no value, as in a finish.
    """
    return {name: None if _is_empty(arguments.get(name)) else arguments[name] for name in names}


def unserved_request(arguments: Mapping[str, Any]) -> str:
    """Return the request a finish hands back, stripped, or ""."""
    unserved = arguments.get(UNSERVED_FIELD)
    return unserved.strip() if isinstance(unserved, str) else ""


def saves_values(arguments: Mapping[str, Any]) -> bool:
    """Return True for a finish that saves a value and hands nothing back."""
    if unserved_request(arguments):
        return False
    return any(not _is_empty(value) for name, value in arguments.items() if name != UNSERVED_FIELD)


def hand_back_keeps_values(*, caller_spoke: bool, action_unrecorded: bool) -> bool:
    """Return whether a hand-back saves the values given with it (``record`` mode).

    Only once the caller has said something in the step, so the values can be what they
    confirmed, and only when no closing tool under the finish is left without a result.
    """
    return caller_spoke and not action_unrecorded


def refusal_key(message: str, arguments: Mapping[str, Any]) -> tuple[str, str]:
    """Return a key that tells a finish resending the values just refused."""
    return message, json.dumps(arguments, sort_keys=True, default=str)


def derived_from(task: TaskSpec, row: Assign) -> str | None:
    """Return the finish parameter a nested value is read from, or None (``TaskSpec.derived_from``)."""
    return task.derived_from(row)


def derived_arguments(task: TaskSpec) -> set[str]:
    """Return the finish parameters a task reads from a parent (``TaskSpec.derived_arguments``)."""
    return task.derived_arguments()


def with_derived_values(task: TaskSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Return the finish arguments with every nested value read from its parent."""
    return dict(arguments) | {
        row.argument: read_path(json_text(arguments.get(parent)), row.path[1:])
        for row in task.assign
        if (parent := task.derived_from(row)) is not None
    }


def values_from_result(
    task: TaskSpec, result: Mapping[str, Any], given: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the finish arguments a closing tool's call carries (``TaskSpec.values_from_result``)."""
    return task.values_from_result(result, given)


def success_matches(rule: FinishRule, result: object) -> bool:
    """Return True when every field of the rule holds one of its success values."""
    return result_matches(rule.success, result)


def _appended(entries: list[Any], entry: object) -> list[Any]:
    """Add one entry onto a list, unless an identical object is already on it."""
    if isinstance(entry, dict | list) and entry in entries:
        return list(entries)
    return [*entries, entry]


def _field_adapter(model: type[StepResult], name: str) -> TypeAdapter[Any]:
    """Return the adapter that validates one finish value as the model's field does."""
    FinishValue = TypedDict("FinishValue", {name: _field_type(model, safe_field_name(name))})  # ty: ignore[invalid-argument-type, invalid-type-form]
    return TypeAdapter(FinishValue)


class StepValues:
    """The values one step saves: checking them, merging a tool's own, and the one-batch save.

    It holds one adapter per value, built once, so a finish checks a value without building
    the whole result model.
    """

    def __init__(self, task: TaskSpec, model: type[StepResult]) -> None:
        """Build the adapters of the task's values.

        Args:
            task: The task.
            model: The step's result model (``result_model``).
        """
        self.task = task
        self.model = model
        self._adapters = {row.argument: _field_adapter(model, row.argument) for row in task.assign}

    def checked(self, name: str, raw: object) -> object:
        """Validate one finish value and return it as JSON data.

        Raises:
            TaskValueRefused: Naming the parameter and what was wrong.
        """
        adapter = self._adapters[name]
        try:
            value = adapter.validate_python({name: raw})
        except ValidationError as error:  # the model reads TaskValueRefused, not a ValidationError
            raise TaskValueRefused(StateRefused.from_error(error).message) from None
        return adapter.dump_python(value, mode="json")[name]

    def _fits(self, name: str, raw: object) -> bool:
        """Return True when the value passes ``checked``."""
        try:
            self.checked(name, raw)
        except TaskValueRefused:
            return False
        return True

    def merge_retained(
        self, arguments: Mapping[str, Any], retained: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Put a closing tool's own valid values back into the model's repair of its save.

        The model cannot make up a booking reference, so a value the tool returned validly wins.
        """
        return dict(arguments) | {
            name: value
            for name, value in self.task.values_from_result(retained).items()
            if value is not None and self._fits(name, value)
        }

    def changed_values(self, arguments: Mapping[str, Any], state: CallState) -> list[str]:
        """Return the finish parameters whose values would change what State holds.

        A value equal to the one held, or one that does not fit (the save refuses it anyway),
        changes nothing here.
        """
        arguments = with_derived_values(self.task, arguments)
        return [
            row.argument
            for row, value in self._fitting(arguments)
            if self._differs(row, value, state)
        ]

    def _fitting(self, arguments: Mapping[str, Any]) -> Iterator[tuple[Assign, object]]:
        """Yield each row whose argument is given and fits, with the value as JSON data."""
        for row in self.task.assign:
            if _is_empty(raw := arguments.get(row.argument)):
                continue
            try:
                yield row, self.checked(row.argument, raw)
            except TaskValueRefused:
                continue

    @staticmethod
    def _differs(row: Assign, value: object, state: CallState) -> bool:
        """Return True when saving ``value`` would change what State holds for the row."""
        held = to_jsonable_python(state.plain(row.variable))
        return value not in held if row.append and isinstance(held, list) else value != held

    def _save(
        self,
        arguments: Mapping[str, Any],
        state: CallState,
        *,
        only: set[str] | None = None,
        required: frozenset[str] = frozenset(),
        keep_valid: bool = False,
        confirming: bool = True,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Save the step's values, or only those named: what saved, and what did not.

        All or none, refused with ``TaskValueRefused``. A value the task needs is refused
        when missing: every row not optional, or with ``only`` the ``required`` ones. With
        ``keep_valid`` a value that does not fit is left out and named, and the rest are
        saved. Without ``confirming`` the step confirms none of them.
        """
        task = self.task
        pending: dict[str, Any] = {}
        refused: list[str] = []
        for row in task.assign:
            if only is not None and row.argument not in only:
                continue
            try:
                value = self.checked(row.argument, arguments.get(row.argument))
            except TaskValueRefused:
                if not keep_valid:
                    raise
                refused.append(row.variable)
                continue
            needed = row.argument in required if only is not None else not row.optional
            if value is None and needed and not row.append:
                raise TaskValueRefused(f"{row.argument}: Field required")
            if row.append:
                current = state.plain(row.variable) or []
                entries = current if value is None else _appended(current, value)
                if entries == current and state.is_confirmed(row.variable):
                    continue  # nothing new onto a list already agreed to is not a write
                value = entries
            pending[row.variable] = value
        try:
            state.save(
                pending,
                step=task.name if confirming else None,
                origin=StateOrigin("step", task.name),
            )
        except StateRefused as refusal:
            if not keep_valid:
                raise TaskValueRefused(refusal.message) from None
            return (), (*refused, *pending)
        saved = tuple(name for name, value in pending.items() if value not in (None, "", []))
        return saved, tuple(refused)

    def save(
        self,
        arguments: Mapping[str, Any],
        *,
        state: CallState,
        hand_back_saves: bool = False,
        confirming: bool = True,
    ) -> SaveOutcome:
        """Check every value of a finish, then save them together, or refuse them all.

        A value left out is saved as empty, and refused when the row is not optional. A
        non-empty ``unserved_request`` hands the caller back and saves nothing, unless
        ``hand_back_saves``: then the values given with it are saved, any that do not fit are
        named in ``refused``, and a hand-back is never refused.

        Args:
            arguments: The finish arguments, or the values a closing tool returned.
            state: The call state.
            hand_back_saves: Save the values given with a hand-back (``record`` mode).
            confirming: The step's save confirms the values it confirms. False for a save
                the call's end makes, which nobody was left to agree to.

        Returns:
            How the step ended.

        Raises:
            TaskValueRefused: If a value does not fit. Nothing was saved.
        """
        arguments = with_derived_values(self.task, arguments)
        if unserved := unserved_request(arguments):
            given = {
                row.argument
                for row in self.task.assign
                if not _is_empty(arguments.get(row.argument))
            }
            if not hand_back_saves or not given:
                return SaveOutcome("unserved", unserved_request=unserved)
            saved, refused = self._save(arguments, state, only=given, keep_valid=True)
            return SaveOutcome("unserved", saved, unserved, refused)
        saved, _ = self._save(arguments, state, confirming=confirming)
        return SaveOutcome("completed", saved)

    def save_rows(
        self,
        rows: Sequence[Assign],
        arguments: Mapping[str, Any],
        *,
        state: CallState,
        status: Literal["completed", "failed"],
    ) -> SaveOutcome:
        """Save only the values a finish rule's own rows name, and end the step with ``status``.

        Each row is required unless it is optional or appends, as in a finish.

        Args:
            rows: The rule's rows. Each names a value the task's ``assign`` declares.
            arguments: Their values, by the task's finish parameters
                (``TaskSpec.values_from_result`` with ``rows``).
            state: The call state.
            status: How the step ends.

        Raises:
            TaskValueRefused: If a value does not fit. Nothing was saved.
        """
        names = {row.variable: row.argument for row in self.task.assign}
        required = frozenset(names[row.variable] for row in rows if not row.optional)
        only = {names[row.variable] for row in rows}
        saved, _ = self._save(arguments, state, only=only, required=required)
        return SaveOutcome(status, saved)


def merge_retained(
    task: TaskSpec,
    arguments: Mapping[str, Any],
    retained: Mapping[str, Any],
    model: type[StepResult],
) -> dict[str, Any]:
    """Put a closing tool's own valid values back into the model's repair of its save."""
    return StepValues(task, model).merge_retained(arguments, retained)


def changed_values(
    task: TaskSpec, arguments: Mapping[str, Any], state: CallState, model: type[StepResult]
) -> list[str]:
    """Return the finish parameters whose values would change what State holds."""
    return StepValues(task, model).changed_values(arguments, state)


def save_task_values(
    task: TaskSpec,
    arguments: Mapping[str, Any],
    *,
    state: CallState,
    model: type[StepResult],
    hand_back_saves: bool = False,
    confirming: bool = True,
) -> SaveOutcome:
    """Check every value of a finish, then save them together, or refuse them all.

    See ``StepValues.save``.

    Args:
        task: The task.
        arguments: The finish arguments, or the values a closing tool returned.
        state: The call state.
        model: The step's result model.
        hand_back_saves: Save the values given with a hand-back (``record`` mode).
        confirming: The step's save confirms the values it confirms.

    Returns:
        How the step ended.

    Raises:
        TaskValueRefused: If a value does not fit. Nothing was saved.
    """
    return StepValues(task, model).save(
        arguments, state=state, hand_back_saves=hand_back_saves, confirming=confirming
    )


def appends_to_list(state: type[BaseModel], row: Assign) -> bool:
    """Return True when the State value ``row`` appends to is a list."""
    return item_type(_field_type(state, state_fields(state)[row.variable])) is not None
