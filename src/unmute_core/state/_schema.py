"""Build a State model from declared variables and shapes, each typed with type text.

An agent defined as data declares its values and its shapes as rows. ``StateSchema`` checks
them and builds the Pydantic class that ``CallState`` runs on, so such code needs no parser
of its own. An Unmute author writes the class by hand instead. Both run through the same
``CallState``.
"""

import functools
import re
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    create_model,
    model_validator,
)

from unmute_core.state._call_info import CALL, reserved_name_error
from unmute_core.state._call_state import CallStartRule
from unmute_core.state._grammar import (
    BUILTIN_SHAPES,
    TypeExpr,
    TypeTextError,
    first_repeat,
    parse_type,
    reserved_field_name,
    reserved_shape_name,
    safe_field_name,
)
from unmute_core.state._types import Date, Id, NameEmail, Phone, Text, Time, json_text

SHAPE_FIELD_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
"""A value or shape name: a letter or underscore, then letters, digits and underscores."""


class StateSchemaError(ValueError):
    """Declared variables or shapes that cannot build a State. The message names the row."""


def _blank_is_none(value: object) -> object:
    """Read empty text as no value yet, never as a wrong one."""
    return None if isinstance(value, str) and not value.strip() else value


_SCALARS: Mapping[str, Any] = MappingProxyType(
    {
        "str": Text,
        "int": int,
        "float": Annotated[float, Field(allow_inf_nan=False)],
        "bool": bool,
        "Phone": Phone,
        "Date": Date,
        "Time": Time,
        "Id": Id,
        "EmailStr": EmailStr,
    }
)


def _canonical(value: str) -> str:
    """Parse type text and return it written the canonical way."""
    try:
        return parse_type(value).canonical
    except TypeTextError as error:
        raise ValueError(str(error)) from None


TypeText = Annotated[str, Field(max_length=8000), AfterValidator(_canonical)]
"""Type text, stored canonical."""


def _choice_of(choices: tuple[str, ...]) -> Callable[[object], object]:
    """Return a check that maps text to the choice it names, whitespace around both ignored.

    An exact choice stays itself. Otherwise the trimmed text picks the first choice that
    trims to it, so a declared ``" small"`` is saved as declared. A JSON number picks the
    choice its digits spell, as ``str`` stores a number as its digits: ``1`` names ``"1"``.
    """
    trimmed = {choice.strip(): choice for choice in reversed(choices)}

    def choice_of(value: object) -> object:
        """Return the declared choice ``value`` names, or ``value`` for the type to refuse."""
        if isinstance(value, int | float) and not isinstance(value, bool):
            whole = isinstance(value, float) and value.is_integer()
            value = str(int(value)) if whole else str(value)
        if isinstance(value, str) and value not in choices:
            return trimmed.get(value.strip(), value)
        return value

    return choice_of


def _shape_field_name(value: str) -> str:
    """Refuse a name that is not lower snake case, or one Pydantic keeps for itself."""
    if not SHAPE_FIELD_PATTERN.fullmatch(value):
        raise ValueError(
            f"shape field {value!r} is not a name this scope takes. A field name is lower "
            "case letters and digits with single underscores between words, starting "
            "with a letter, like scheduled_date"
        )
    if reason := reserved_field_name(value):
        raise ValueError(
            f"a shape field cannot be called {value!r}: {reason}. Give it another name; "
            "one Pydantic does not use, like model_year, is fine"
        )
    return value


class _Row(BaseModel):
    """Immutable, and refuses keys it does not know."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ShapeField(_Row):
    """One field of a shape.

    Attributes:
        name: Lower snake case, like ``scheduled_date``.
        type: Its type text, stored canonical.
        description: What it holds, said to the model.
    """

    name: Annotated[str, AfterValidator(_shape_field_name)]
    type: TypeText
    description: str = ""


class Shape(_Row):
    """A named group of fields a type may name, like ``Contact``.

    Attributes:
        name: Written in CapWords by convention.
        description: What it holds, said to the model.
        fields: One field or more.
    """

    name: Name
    description: str = ""
    fields: Annotated[tuple[ShapeField, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Refuse a reserved name and a field declared twice."""
        if reason := reserved_shape_name(self.name):
            raise ValueError(
                f"shape {self.name!r} cannot be called that, because {reason}. Give it a name "
                "of its own, written in CapWords"
            )
        if twice := first_repeat(row.name for row in self.fields):
            raise ValueError(f"shape {self.name!r} declares field {twice!r} twice")
        return self


class Variable(_Row):
    """One value of the State.

    Attributes:
        name: The value's name, as prompts and tasks name it.
        description: What it holds, said to the model.
        type: Its type text, stored canonical. Text by default.
        confirm: The task that must hear the caller agree to it.
        call_start: ``required`` or ``optional`` for a value the call is dispatched with.
    """

    name: Name
    description: str = ""
    type: TypeText = "str"
    confirm: Annotated[str | None, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")] = None
    call_start: CallStartRule | None = None


class StateSchema:
    """Declared variables and shapes, checked, and the State class built from them.

    Every value starts the call empty: a list as ``[]``, anything else as None. Empty text
    reads as no value yet. Inside a shape, a field is required unless its type ends in
    ``| None``, and a list field starts empty.

    Attributes:
        variables: The declared values, in order.
        shapes: The declared shapes, in order. ``NameEmail`` is built in.
        types: Each value's parsed type.
        confirm: Value name to the task that confirms it, for ``CallState``.
        call_start: Value name to ``required`` or ``optional``, for ``CallState``.
        model: The State class.
    """

    def __init__(
        self,
        variables: Iterable[Variable | Mapping[str, Any]],
        shapes: Iterable[Shape | Mapping[str, Any]] = (),
        *,
        name: str = "State",
    ) -> None:
        """Check the rows and build the State class.

        Args:
            variables: The values, as ``Variable`` or its JSON.
            shapes: The shapes, as ``Shape`` or its JSON.
            name: The State class name.

        Raises:
            pydantic.ValidationError: If a row is malformed.
            StateSchemaError: If the rows do not fit together, naming the row and the fix.
        """
        self.variables = tuple(Variable.model_validate(row) for row in variables)
        self.shapes = tuple(Shape.model_validate(row) for row in shapes)
        self._check_names()
        self._fields = {
            shape.name: {row.name: self._parse_field(shape, row) for row in shape.fields}
            for shape in self.shapes
        }
        self._check_cycles()
        self.types = {row.name: self._parse(row) for row in self.variables}
        self.confirm = {row.name: row.confirm for row in self.variables if row.confirm}
        self.call_start = {row.name: row.call_start for row in self.variables if row.call_start}
        self._classes: dict[str, type[BaseModel]] = {"NameEmail": NameEmail}
        self.model = self._state_model(name)

    @functools.cached_property
    def _shape_names(self) -> frozenset[str]:
        """Every shape a type may name: the declared ones and the built-in ones."""
        return frozenset(shape.name for shape in self.shapes) | BUILTIN_SHAPES

    @functools.cached_property
    def _declared(self) -> dict[str, Shape]:
        """The declared shapes by name."""
        return {shape.name: shape for shape in self.shapes}

    def _parse_field(self, shape: Shape, row: ShapeField) -> TypeExpr:
        """Parse one shape field's type against the known shapes."""
        try:
            return parse_type(row.type, self._shape_names)
        except TypeTextError as error:
            raise StateSchemaError(f"shape {shape.name!r} field {row.name!r}: {error}") from None

    def _check_names(self) -> None:
        """Refuse a name declared twice, and the library's own ``call``."""
        if any(row.name == CALL for row in self.variables):
            raise StateSchemaError(str(reserved_name_error("runtime variable")))
        if twice := first_repeat(row.name for row in self.variables):
            raise StateSchemaError(f"runtime variable {twice!r} is declared twice")
        if twice := first_repeat(safe_field_name(row.name) for row in self.variables):
            names = sorted(row.name for row in self.variables if safe_field_name(row.name) == twice)
            raise StateSchemaError(
                f"runtime variables {names} would share the field {twice!r}: rename one of them"
            )
        if twice := first_repeat(shape.name for shape in self.shapes):
            raise StateSchemaError(
                f"shape {twice!r} is declared twice. One shape, one name: merge the two or "
                "give the second its own name"
            )

    def _check_cycles(self) -> None:
        """Refuse a shape that contains itself, through any chain of fields."""
        refers = {
            shape: set().union(*(field.shape_names() for field in fields.values()))
            for shape, fields in self._fields.items()
        }
        done: set[str] = set()

        def walk(name: str, stack: tuple[str, ...]) -> None:
            """Follow every shape ``name`` refers to, failing on one already on the stack."""
            if name in stack:
                chain = " -> ".join([*stack[stack.index(name) :], name])
                raise StateSchemaError(
                    f"shape {name!r} refers to itself, through {chain}. A shape that contains "
                    "itself cannot be built: break the chain, or hold the nested part as a "
                    "list on the outer shape"
                )
            if name in done or name not in refers:
                return
            for other in sorted(refers[name]):
                walk(other, (*stack, name))
            done.add(name)

        for name in sorted(refers):
            walk(name, ())

    def _parse(self, row: Variable) -> TypeExpr:
        """Parse one value's type against the known shapes."""
        try:
            return parse_type(row.type, self._shape_names)
        except TypeTextError as error:
            raise StateSchemaError(f"runtime variable {row.name!r}: {error}") from None

    def shape_fields(self, shape: str) -> dict[str, TypeExpr]:
        """Return a shape's fields and their types, the built-in ``NameEmail`` included.

        Raises:
            KeyError: If no such shape is known.
        """
        if shape == "NameEmail":
            return {"name": TypeExpr(base="str"), "email": TypeExpr(base="EmailStr")}
        return self._fields[shape]

    def annotation(self, expr: TypeExpr) -> Any:
        """Return the Pydantic annotation a parsed type validates with."""
        if expr.base == "shape" and expr.shape is not None:
            shape = self.shape(expr.shape)
            annotation: Any = Annotated[shape, BeforeValidator(json_text)]  # ty: ignore[invalid-type-form]
        elif expr.base == "list" and expr.item is not None:
            item = self.annotation(expr.item)
            annotation = Annotated[list[item], BeforeValidator(json_text)]
        elif expr.base == "Literal":
            annotation = Annotated[
                Literal[expr.choices], BeforeValidator(_choice_of(expr.choices))  # ty: ignore[invalid-type-form]
            ]
        else:
            annotation = _SCALARS[expr.base]
        return annotation | None if expr.optional else annotation

    def shape(self, name: str) -> type[BaseModel]:
        """Return the Pydantic class of a declared shape, built once.

        A tool defined as data names it as its output model. A State value of that shape
        then takes the tool's whole result, since both are the same class.

        Args:
            name: The shape's name, as declared.

        Raises:
            KeyError: If no shape has that name.
        """
        if name not in self._classes and name not in self._declared:
            raise KeyError(f"no shape named {name!r}. The shapes are: {', '.join(self._declared)}")
        if name not in self._classes:
            shape = self._declared[name]
            fields: dict[str, Any] = {
                row.name: self._field(self._fields[name][row.name], row.description, inner=True)
                for row in shape.fields
            }
            self._classes[name] = create_model(
                name,
                __config__=ConfigDict(extra="ignore"),
                __doc__=shape.description or None,
                **fields,
            )
        return self._classes[name]

    def _field(
        self, expr: TypeExpr, description: str, *, inner: bool, alias: str | None = None
    ) -> tuple[Any, Any]:
        """Return one field's annotation and default.

        A list starts empty. Inside a shape, any other field is required unless optional.
        At the top of the State every value is optional. Empty text is no value, at the top
        and in an optional field of a shape alike.
        ``alias`` is the real name of a value whose field name is not its own.
        """
        described = description.strip() or None
        annotation = self.annotation(expr)
        if expr.base == "list":
            return annotation, Field(default_factory=list, description=described, alias=alias)
        if inner and not expr.optional:
            return annotation, Field(..., description=described)
        if inner:
            return (
                Annotated[annotation, BeforeValidator(_blank_is_none)],
                Field(None, description=described),
            )
        optional = annotation if expr.optional else annotation | None
        return (
            Annotated[optional, BeforeValidator(_blank_is_none)],
            Field(None, description=described, alias=alias),
        )

    def _state_model(self, name: str) -> type[BaseModel]:
        """Build the State class: one field per value, every one empty at call start."""
        fields: dict[str, Any] = {
            safe_field_name(row.name): self._field(
                self.types[row.name],
                row.description,
                inner=False,
                alias=row.name if safe_field_name(row.name) != row.name else None,
            )
            for row in self.variables
        }
        return create_model(name, **fields)
