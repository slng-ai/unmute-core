"""Paths into the State: ``{{customer.status}}``, also written ``{{customer__status}}``.

Prompts, spoken lines and locked tool arguments name a value, or a field inside a shape
value, the same way. The dotted form is the one authors write. The ``__`` form is
Unmute's, and the SLNG Context Router's, whose placeholder names hold only letters, digits
and underscores. Both read the same value.

A path names at most 8 fields.
"""

import re
import types
from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import Annotated, Any, Self, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors

MAX_PATH_FIELDS = 8
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\s*\}\}")
"""A ``{{name}}``, ``{{name.field}}`` or ``{{name__field}}`` placeholder, spaces allowed."""

_PART = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ValuePathError(ValueError):
    """A path that names nothing in the State. The message says what the State holds."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class ValuePath:
    """A State value, or a field inside a shape value.

    Attributes:
        root: The State value.
        fields: The fields inside it, outermost first. Empty for the whole value.
    """

    root: str
    fields: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The path written with dots, as ``CallState.plain`` reads it: ``customer.status``."""
        return ".".join((self.root, *self.fields))

    @property
    def key(self) -> str:
        """The path as one placeholder name, the router's form: ``customer__status``."""
        return "__".join((self.root, *self.fields))

    @classmethod
    def parse(cls, name: str, names: Collection[str] = ()) -> Self:
        """Read a placeholder name as a path into the State.

        A name with dots splits on the dots. Otherwise a whole name the State holds is that
        value, so a field called ``a__b`` reads as itself, and any other name splits on ``__``.

        Args:
            name: The placeholder's name, like ``customer.status`` or ``customer__status``.
            names: The State's value names.

        Returns:
            The path.

        Raises:
            ValuePathError: If the path mixes the two forms, has an empty part, or is deeper
                than 8 fields.
        """
        if "." in name and "__" in name:
            raise ValuePathError(
                f"{{{{{name}}}}} mixes dots and double underscores. Write {{{{{name.replace('__', '.')}}}}}"
            )
        if "." not in name and name in names:
            return cls(name)
        parts = name.split(".") if "." in name else name.split("__")
        if not all(_PART.fullmatch(part) for part in parts):
            raise ValuePathError(
                f"{{{{{name}}}}} is not a value name or a path like {{{{name.field}}}}"
            )
        if len(parts) - 1 > MAX_PATH_FIELDS:
            raise ValuePathError(
                f"{{{{{name}}}}} names {len(parts) - 1} fields deep, and a path names at most "
                f"{MAX_PATH_FIELDS}"
            )
        return cls(parts[0], tuple(parts[1:]))

    def annotation(self, model: type[BaseModel]) -> Any:
        """Return the annotation this path names in a State model.

        A path may name a whole shape or a whole list. It cannot name a field inside a list,
        because nothing says which entry it means.

        Args:
            model: The State model: an author's own class, or one ``StateSchema`` built.

        Returns:
            The annotation of the value or field the path names.

        Raises:
            ValuePathError: If the path names nothing, naming what the State or shape holds.
        """
        fields = _field_infos(model)
        if self.root not in fields:
            raise ValuePathError(
                f"names no value of the State: {self.root!r}. It holds {', '.join(fields)}"
            )
        annotation: Any = fields[self.root].annotation
        walked = [self.root]
        for part in self.fields:
            where = ".".join(walked)
            if is_list(annotation):
                raise ValuePathError(
                    f"{where} is a list, and a path cannot name a field inside one: nothing "
                    "says which entry it means"
                )
            shape = _model_class(annotation)
            if shape is None:
                raise ValuePathError(
                    f"{where} is {_type_name(annotation)}, which has no fields to name"
                )
            inner = _field_infos(shape)
            if part not in inner:
                raise ValuePathError(
                    f"shape {shape.__name__!r} declares no field {part!r}. It declares "
                    f"{', '.join(inner)}"
                )
            annotation = inner[part].annotation
            walked.append(part)
        return annotation


def parse_path(name: str, names: Collection[str] = ()) -> ValuePath:
    """Read a placeholder name as a path into the State: ``ValuePath.parse``."""
    return ValuePath.parse(name, names)


def path_annotation(model: type[BaseModel], path: ValuePath) -> Any:
    """Return the annotation a path names in a State model: ``ValuePath.annotation``."""
    return path.annotation(model)


def _field_infos(model: type[BaseModel]) -> dict[str, Any]:
    """Return a model's fields by the name a path uses: the alias, else the field name."""
    return {info.alias or name: info for name, info in model.model_fields.items()}


def _unwrapped(annotation: Any) -> Any:
    """Strip ``Annotated`` and ``| None`` off an annotation."""
    while True:
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        elif get_origin(annotation) in (Union, types.UnionType):
            real = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(real) != 1:
                return annotation
            annotation = real[0]
        else:
            return annotation


def _model_class(annotation: Any) -> type[BaseModel] | None:
    """Return the Pydantic model an annotation holds, or None."""
    found = _unwrapped(annotation)
    return found if isinstance(found, type) and issubclass(found, BaseModel) else None


def is_list(annotation: Any) -> bool:
    """Return True when an annotation holds a list."""
    return get_origin(_unwrapped(annotation)) is list


def is_single_value(annotation: Any) -> bool:
    """Return True when an annotation holds one plain value: not a shape, not a list."""
    return _model_class(annotation) is None and not is_list(annotation)


_FORMAT_BASES = MappingProxyType(
    {
        "phone": "Phone",
        "date": "Date",
        "birth-date": "Date",
        "time": "Time",
        "id": "Id",
        "email": "EmailStr",
    }
)
_TYPE_BASES = MappingProxyType(
    {"integer": "int", "number": "float", "boolean": "bool", "string": "str"}
)


def _scalar_schema(annotation: Any) -> Mapping[str, Any]:
    """Return the JSON Schema of a single value, ``| None`` left out, a ``const`` as an enum.

    Not cached: a shape type is a class built per call, and a cache would keep it alive.
    It runs only when tools and prefetch entries are built.
    """
    schema = TypeAdapter(annotation).json_schema()
    real = [node for node in schema.get("anyOf", [schema]) if node.get("type") != "null"]
    node = real[0] if len(real) == 1 else {}
    return node | {"enum": [node["const"]]} if "const" in node else node


def value_base(annotation: Any) -> str:
    """Name what a State value holds, as the type grammar names it.

    One of ``str``, ``int``, ``float``, ``bool``, ``Phone``, ``Date``, ``Time``, ``Id``,
    ``EmailStr``, ``Literal``, ``shape`` and ``list``. It reads any Pydantic annotation, so
    an author's own ``datetime.date`` is a ``Date`` too.
    """
    if _model_class(annotation) is not None:
        return "shape"
    if is_list(annotation):
        return "list"
    node = _scalar_schema(annotation)
    if "enum" in node:
        return "Literal"
    if node.get("format") in _FORMAT_BASES:
        return _FORMAT_BASES[node["format"]]
    return _TYPE_BASES.get(str(node.get("type")), "str")


def literal_choices(annotation: Any) -> tuple[object, ...]:
    """Return the values a ``Literal`` State value offers, or () for any other value."""
    return tuple(_scalar_schema(annotation).get("enum", ()))


def _type_name(annotation: Any) -> str:
    """Name an annotation for an error message."""
    found = _unwrapped(annotation)
    return getattr(found, "__name__", None) or str(found)
