"""The type grammar: the type text a variable or a shape field declares, like ``Phone | None``.

The grammar is Unmute's, and every reader of an agent's types parses the same texts:
``type := atom ("|" atom)*``, ``atom := name | name "[" args "]"``, ``arg := type | "quoted"``.
It takes text, ``int``, ``float``, ``bool``, the value types ``Phone``, ``Date``, ``Time``,
``Id`` and ``EmailStr``, ``Literal[...]``, ``list[T]``, named shapes and the built-in
``NameEmail``, each optionally ``| None``.

The refusal wording is part of the contract: tests match on it.
"""

import json
import re
from collections.abc import Collection, Hashable, Iterable, Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors

TypeBase = Literal[
    "str",
    "int",
    "float",
    "bool",
    "Phone",
    "Date",
    "Time",
    "Id",
    "EmailStr",
    "Literal",
    "shape",
    "list",
]
"""What a type is, before ``| None``."""

TYPE_ALIASES: Mapping[str, TypeBase] = MappingProxyType(
    {
        "str": "str",
        "string": "str",
        "int": "int",
        "integer": "int",
        "float": "float",
        "number": "float",
        "bool": "bool",
        "boolean": "bool",
    }
)
VALUE_TYPES = ("Phone", "Date", "Time", "Id", "EmailStr")
BUILTIN_SHAPES = frozenset({"NameEmail"})
MAX_TYPE_DEPTH = 16
"""Brackets a type may nest. Nothing valid nests past ``list[Literal[...]]``, and the bound
keeps a long type text from overflowing the parser's stack."""
MAX_TYPE_TEXT = 8000

REFUSED_TYPES: Mapping[str, str] = MappingProxyType(
    {
        "datetime": (
            'write "Date" and "Time" as two fields, which are text with a shape checked where the '
            "value enters the state and never written into the schema the model is sent"
        ),
        "date": 'write "Date", which is text with a validated shape',
        "time": 'write "Time", which is text with a validated shape',
        "UUID": 'write "Id", which is text with a validated shape',
        "email": (
            'write "EmailStr", which is text with a validated shape, or "NameEmail" to hold a '
            "name beside the address"
        ),
        "Email": (
            'write "EmailStr", which is text with a validated shape, or "NameEmail" to hold a '
            "name beside the address"
        ),
        "SecretStr": (
            "a secret never travels through state: it reaches a tool through that tool's own "
            "secret field"
        ),
        "PaymentCardNumber": (
            "card numbers are outside the declared type scope, and nothing here should carry one"
        ),
        "dict": (
            'declare the fields as a shape under "shapes:" and name that shape here, or name one '
            "of the built-in shapes"
        ),
        "Dict": (
            'declare the fields as a shape under "shapes:" and name that shape here, or name one '
            "of the built-in shapes"
        ),
        "set": 'write "list[...]", which is the one collection this scope takes',
        "tuple": 'write "list[...]", which is the one collection this scope takes',
        "List": 'write "list[...]", in lower case, which is Python\'s own spelling now',
        "Optional": 'write "T | None"',
        "Union": 'write "A | B", and only "| None" is meaningful in this scope',
        "Any": (
            "name the type: a value nothing can describe is a value no prompt can render and no "
            "guard can test"
        ),
        "BaseModel": 'name one of the shapes declared under "shapes:", or one of the built-in shapes',
    }
)
"""What a person reaches for, and what to write instead."""

_TOKEN = re.compile(
    r"""\s*(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<string>"(?:[^"\\]|\\.)*"|'[^'\\]*')"""
    r"""|(?P<punct>[\[\],|])|(?P<number>[-+]?\d+(?:\.\d+)?))"""
)


class TypeTextError(ValueError):
    """A type text the grammar does not take. The message says what to write instead."""


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class TypeExpr:
    """One parsed type.

    Attributes:
        base: What it is, before ``| None``.
        choices: The entries of a ``Literal``.
        optional: It ends in ``| None``.
        shape: The shape a ``shape`` names.
        item: The entry type of a ``list``.
    """

    base: TypeBase
    choices: tuple[str, ...] = ()
    optional: bool = False
    shape: str | None = None
    item: "TypeExpr | None" = None

    @property
    def canonical(self) -> str:
        """The type written the one way every reader writes it, like ``list[Contact]``."""
        if self.base == "Literal":
            text = f"Literal[{', '.join(json.dumps(c, ensure_ascii=False) for c in self.choices)}]"
        elif self.base == "list" and self.item is not None:
            text = f"list[{self.item.canonical}]"
        elif self.base == "shape" and self.shape is not None:
            text = self.shape
        else:
            text = self.base
        return f"{text} | None" if self.optional else text

    @property
    def structured(self) -> bool:
        """True for a shape or a list: a value with parts."""
        return self.base in ("shape", "list")

    def shape_names(self) -> set[str]:
        """Return every shape this type names, through a list."""
        if self.base == "shape" and self.shape is not None:
            return {self.shape}
        if self.base == "list" and self.item is not None:
            return self.item.shape_names()
        return set()


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class _Atom:
    """One name, quoted entry or bracketed name, before it is resolved to a type."""

    name: str = ""
    entry: str | None = None
    args: tuple[tuple["_Atom", ...], ...] = ()
    bracket: bool = False

    def text(self) -> str:
        """Write the atom back as type text, for an error message."""
        if self.entry is not None:
            return json.dumps(self.entry, ensure_ascii=False)
        if not self.bracket:
            return self.name
        inner = ", ".join(" | ".join(atom.text() for atom in arg) for arg in self.args)
        return f"{self.name}[{inner}]"


def first_repeat[T: Hashable](items: Iterable[T]) -> T | None:
    """Return the first item that has already appeared, or None when all are distinct."""
    seen: set[T] = set()
    for item in items:
        if item in seen:
            return item
        seen.add(item)
    return None


def _tokens(text: str) -> list[tuple[str, str]]:
    """Split type text into (kind, value) tokens. A quoted entry is unquoted."""
    out: list[tuple[str, str]] = []
    position = 0
    while text[position:].strip():
        match = _TOKEN.match(text, position)
        if match is None:
            raise TypeTextError(f"cannot read the type at {text[position:].strip()!r}")
        kind = match.lastgroup or ""
        value = match.group(kind)
        if kind == "string":
            value = json.loads(value) if value.startswith('"') else value[1:-1]
        out.append((kind, value))
        position = match.end()
    return out


class _Parser:
    """Reads the tokens of one type text, one atom at a time."""

    def __init__(self, text: str) -> None:
        """Tokenize ``text``."""
        self.tokens = _tokens(text)
        self.at = 0
        self.depth = 0

    def peek(self) -> tuple[str, str] | None:
        """Return the next token without taking it, or None at the end."""
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        """Take the next token."""
        if (token := self.peek()) is None:
            raise TypeTextError("the type ends too early")
        self.at += 1
        return token

    def union(self) -> tuple[_Atom, ...]:
        """Read atoms joined by ``|``."""
        atoms = [self.atom()]
        while self.peek() == ("punct", "|"):
            self.take()
            atoms.append(self.atom())
        return tuple(atoms)

    def atom(self) -> _Atom:
        """Read a name, a quoted entry, or a name with bracketed arguments."""
        kind, value = self.take()
        if kind == "string":
            return _Atom(entry=value)
        if kind == "number":
            return _Atom(name=value)
        if kind != "name":
            raise TypeTextError(f"expected a type name, found {value!r}")
        if self.peek() != ("punct", "["):
            return _Atom(name=value)
        self.take()
        self.depth += 1
        if self.depth > MAX_TYPE_DEPTH:
            raise TypeTextError(
                f"the type nests more than {MAX_TYPE_DEPTH} brackets deep. "
                "Nothing valid nests past list[Literal[...]]"
            )
        args: list[tuple[_Atom, ...]] = []
        if self.peek() != ("punct", "]"):
            args.append(self.union())
            while self.peek() == ("punct", ","):
                self.take()
                args.append(self.union())
        if self.take() != ("punct", "]"):
            raise TypeTextError(f'"{value}[" is not closed')
        self.depth -= 1
        return _Atom(name=value, args=tuple(args), bracket=True)


def _resolve(union: tuple[_Atom, ...], shapes: Collection[str] | None) -> TypeExpr:
    """Resolve a union of atoms: one real type, and at most one ``None``."""
    nones = [atom for atom in union if atom.entry is None and atom.name == "None"]
    real = [atom for atom in union if atom not in nones]
    if any(atom.bracket for atom in nones):
        raise TypeTextError('"None" takes no brackets')
    if len(nones) > 1:
        raise TypeTextError('"None" written twice. One "| None" is what says a value may be absent')
    if not real:
        raise TypeTextError(
            '"None" and nothing else. A value that can only be absent is not a value: name '
            "the type it holds when it is present"
        )
    if len(real) > 1:
        names = " and ".join(atom.text() for atom in real)
        raise TypeTextError(
            f'a union of {names}. Only "| None" is meaningful in this scope: a value holding '
            "either of two real types cannot be rendered into a prompt or tested by a guard"
        )
    resolved = _resolve_atom(real[0], shapes)
    if nones and resolved.base == "list":
        raise TypeTextError(
            "a list that may be absent. A declared list starts empty, so it is never absent: "
            'drop the "| None", and an empty list is how the state says nothing has been '
            "recorded"
        )
    return TypeExpr(
        base=resolved.base,
        choices=resolved.choices,
        optional=bool(nones),
        shape=resolved.shape,
        item=resolved.item,
    )


def _literal(atom: _Atom) -> TypeExpr:
    """Resolve ``Literal[...]``: quoted, distinct entries of 1 to 100 characters."""
    if not atom.bracket or not atom.args:
        raise TypeTextError(
            "Literal with no entries. A Literal is a closed set, so name at least one: "
            'Literal["yes", "no"]'
        )
    # the refusal wording is part of TypeTextError's public text, not a field constraint
    if bad := [arg for arg in atom.args if len(arg) != 1 or arg[0].entry is None]:
        text = " | ".join(part.text() for part in bad[0])
        raise TypeTextError(
            f"{text} is not a quoted entry. Every entry of a Literal is text in double quotes"
        )
    entries = [str(arg[0].entry) for arg in atom.args]
    if (repeated := first_repeat(entries)) is not None:
        raise TypeTextError(
            f"the entry {json.dumps(repeated)} twice. A Literal is a set, so a repeat says "
            "nothing the first did not"
        )
    if not all(entries):
        raise TypeTextError("a Literal choice must not be empty")
    return TypeExpr(base="Literal", choices=tuple(entries))


def _list(atom: _Atom, shapes: Collection[str] | None) -> TypeExpr:
    """Resolve ``list[T]``: one entry type, never a list itself."""
    if not atom.bracket or not atom.args:
        raise TypeTextError(
            "a list with no element type. Name what it holds: list[Appointment] or "
            'list[Literal["yes", "no"]]'
        )
    if len(atom.args) > 1:
        raise TypeTextError(
            f"a list of {len(atom.args)} types. A list holds one type: list[Appointment]"
        )
    element = _resolve(atom.args[0], shapes)
    if element.base == "list":
        raise TypeTextError(
            "a list of lists. Nothing renders that into a prompt a step can read: declare a "
            "shape holding the inner list and make it a list of that"
        )
    return TypeExpr(base="list", item=element)


def _resolve_atom(atom: _Atom, shapes: Collection[str] | None) -> TypeExpr:
    """Resolve one atom to a type, refusing what the grammar does not take."""
    if atom.entry is not None:
        raise TypeTextError(
            f"the entry {atom.text()} where a type belongs. A quoted entry belongs inside "
            'Literal["a", "b"]'
        )
    name = atom.name
    if name == "Literal":
        return _literal(atom)
    if name == "list":
        return _list(atom, shapes)
    if name in REFUSED_TYPES:
        raise TypeTextError(f"{name!r} is not a type the grammar takes: {REFUSED_TYPES[name]}")
    if atom.bracket:
        raise TypeTextError(f'"{name}" takes no brackets. Only "list" and "Literal" do')
    if name in TYPE_ALIASES:
        return TypeExpr(base=TYPE_ALIASES[name])
    if name in VALUE_TYPES:
        return TypeExpr(base=name)
    if shapes is None or name in BUILTIN_SHAPES or name in shapes:
        return TypeExpr(base="shape", shape=name)
    vocabulary = [
        "str",
        "int",
        "float",
        "bool",
        *VALUE_TYPES,
        *sorted(BUILTIN_SHAPES),
        "Literal[...]",
        "list[...]",
        *sorted(shapes),
    ]
    raise TypeTextError(
        f'no shape named {name!r} is declared. Declare it under "shapes:", or write one of '
        f"{', '.join(vocabulary)}"
    )


def parse_type(text: str, shapes: Collection[str] | None = None) -> TypeExpr:
    """Parse type text.

    Args:
        text: The type, like ``list[Contact]`` or ``Literal["open", "closed"] | None``.
        shapes: The declared shape names. None takes any other name as a shape to resolve
            later, which is how one row is checked before the document's shapes are known.

    Returns:
        The parsed type.

    Raises:
        TypeTextError: If the grammar does not take the text. The message says what to write.
    """
    if len(text) > MAX_TYPE_TEXT:
        raise TypeTextError(f"the type is longer than {MAX_TYPE_TEXT} characters")
    parser = _Parser(text.strip())
    union = parser.union()
    if (extra := parser.peek()) is not None:
        raise TypeTextError(f"unexpected {extra[1]!r} after the type")
    return _resolve(union, shapes)


def reserved_field_name(name: str) -> str:
    """Return why a shape field cannot take ``name``, or "" when it is free.

    Shapes are Pydantic classes, and Pydantic keeps the names of its own model API: some
    break the class as it is built, and ``model_config`` silently drops the field. A name
    like ``model_year`` is free.
    """
    if name.startswith("model_") and hasattr(BaseModel, name):
        return "Pydantic, which checks shapes, uses that name for itself"
    return ""


def safe_field_name(name: str) -> str:
    """Return the Pydantic field name that carries a variable called ``name``.

    Pydantic drops a field with a leading underscore and breaks on a ``model_*`` name that
    ``BaseModel`` owns. Such a name is stored as ``v__<name>`` and keeps its real name as the
    field's alias. Any other name is its own field name.
    """
    if name.startswith("_") or (name.startswith("model_") and hasattr(BaseModel, name)):
        return f"v__{name}"
    return name


def reserved_shape_name(name: str) -> str:
    """Return why a shape cannot take ``name``, or "" when it is free."""
    if name in TYPE_ALIASES:
        return "it is one of the primitive types"
    if name in VALUE_TYPES:
        return "it is one of the text types with a validated shape"
    if name in BUILTIN_SHAPES:
        return "it is one of the built-in shapes"
    if name in REFUSED_TYPES:
        return "a type expression cannot use that name"
    if name in {"Literal", "list", "None"}:
        return "it is part of the type grammar"
    return ""
