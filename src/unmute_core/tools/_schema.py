"""The JSON Schema subset a tool's parameters may use, and the schema the model sees.

The subset is JSON Schema 2020-12 with a fixed list of keywords and limits on size, depth
and branches. A schema outside it fails when the tool is built, with a message that names
what to fix.

The model sees a plainer copy (``model_shape``): refs inlined, ``format`` and ``pattern``
as words, and no null branch on an optional value. Small models fill a plain schema more
reliably; Gemini Flash-Lite answers a complex one with a malformed function call.
"""

import functools
import json
import math
from collections.abc import Collection
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

MAX_SCHEMA_BYTES = 65_536
MAX_SCHEMA_DEPTH = 32
MAX_SCHEMA_NODES = 2_048
MAX_PROPERTIES = 256
MAX_BRANCHES = 32
MAX_VALUE_LIST = 256
SUPPORTED_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "$defs",
        "$ref",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "enum",
        "const",
        "anyOf",
        "oneOf",
        "allOf",
        "not",
        "description",
        "title",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "format",  # an annotation only: nothing validates it, the model reads it as a hint
        "pattern",
        "$comment",
        "example",
        "minProperties",
        "maxProperties",
        "patternProperties",
        "propertyNames",
        "contains",
        "minContains",
        "maxContains",
        "prefixItems",
        "discriminator",
    }
)
ROOT_KEYWORDS = frozenset(
    {
        "$schema",
        "$defs",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "description",
        "title",
        "$comment",
    }
)
_NUMERIC_KEYWORDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")


class ParameterSchemaError(ValueError):
    """A tool's parameter schema is outside the subset the library accepts."""


class _Walk:
    """One pass over a schema: counts nodes, checks keywords and collects local refs."""

    def __init__(self) -> None:
        """Start with no nodes seen."""
        self.nodes = 0
        self.value_nodes = 0
        self.refs: dict[str, set[str]] = {}

    def value(self, value: Any, path: str, depth: int = 0) -> None:
        """Check a literal like an ``enum`` entry: finite JSON within the node limits."""
        self.value_nodes += 1
        if self.value_nodes > MAX_SCHEMA_NODES:
            raise ParameterSchemaError(
                "schema literal values exceed the node limit. Shorten the enum, default or example lists"
            )
        if depth > MAX_SCHEMA_DEPTH:
            raise ParameterSchemaError(
                "schema literal value exceeds the depth limit. Flatten the nested value"
            )
        if value is None or isinstance(value, str | bool | int):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ParameterSchemaError(f"non-finite number at {path}. Use a finite number")
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                self.value(child, f"{path}[{index}]", depth + 1)
            return
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ParameterSchemaError(
                    f"object keys at {path} must be strings. Quote each key as JSON text"
                )
            for key, child in value.items():
                self.value(child, f"{path}.{key}", depth + 1)
            return
        raise ParameterSchemaError(
            f"unsupported JSON value at {path}. Use text, numbers, booleans, null, lists and objects"
        )

    def node(self, node: Any, path: str, depth: int, owner: str) -> None:
        """Check one schema node and walk into its children."""
        self._admit(node, path, depth)
        self._literals(node, path)
        self._finite_numbers(node, path)
        properties = self._mapping(node, "properties", path)
        if len(properties) > MAX_PROPERTIES:
            raise ParameterSchemaError("parameter schema exceeds the property limit")
        definitions = self._mapping(node, "$defs", path)
        self._reference(node, owner)
        self._branches(node, path, depth, owner)
        for name, child in properties.items():
            self.node(child, f"{path}.properties.{name}", depth + 1, owner)
        for name, child in self._mapping(node, "patternProperties", path).items():
            self.node(child, f"{path}.patternProperties.{name}", depth + 1, owner)
        for name, child in definitions.items():
            self.node(child, f"{path}.$defs.{name}", depth + 1, name)
        self._single_schemas(node, path, depth, owner)

    def _admit(self, node: Any, path: str, depth: int) -> None:
        """Count the node against the budgets, and check its shape, keywords and dialect."""
        self.nodes += 1
        if self.nodes > MAX_SCHEMA_NODES:
            raise ParameterSchemaError("parameter schema exceeds the node limit")
        if depth > MAX_SCHEMA_DEPTH:
            raise ParameterSchemaError("parameter schema exceeds the depth limit")
        if not isinstance(node, dict):
            raise ParameterSchemaError(f"schema node at {path} must be an object")
        if "definitions" in node:
            raise ParameterSchemaError(f"definitions at {path} is not supported. Use $defs")
        if "nullable" in node:
            raise ParameterSchemaError(
                f'nullable at {path} is not JSON Schema. Write "type": ["<type>", "null"] instead.'
            )
        if unknown := {key for key in node if not key.startswith("x-")} - SUPPORTED_KEYWORDS:
            raise ParameterSchemaError(f"unsupported schema keyword at {path}: {min(unknown)}")
        dialect = node.get("$schema")
        if dialect is not None and (path != "$" or dialect != SUPPORTED_DIALECT):
            raise ParameterSchemaError(f"$schema must be exactly {SUPPORTED_DIALECT} at the root")

    def _literals(self, node: dict[str, Any], path: str) -> None:
        """Check the list keywords and the ``const`` and ``default`` literals."""
        for keyword in ("required", "enum", "examples"):
            values = node.get(keyword)
            if values is not None:
                if not isinstance(values, list):
                    raise ParameterSchemaError(f"{keyword} at {path} must be an array")
                if len(values) > MAX_VALUE_LIST:
                    raise ParameterSchemaError(
                        f"{keyword} at {path} exceeds the {MAX_VALUE_LIST}-item limit"
                    )
                self.value(values, f"{path}.{keyword}")
        for keyword in ("const", "default"):
            if keyword in node:
                self.value(node[keyword], f"{path}.{keyword}")

    @staticmethod
    def _finite_numbers(node: dict[str, Any], path: str) -> None:
        """Refuse an infinite or NaN numeric keyword."""
        for keyword in _NUMERIC_KEYWORDS:
            value = node.get(keyword)
            if isinstance(value, float) and not math.isfinite(value):
                raise ParameterSchemaError(f"{keyword} at {path} must be finite")

    @staticmethod
    def _mapping(node: dict[str, Any], keyword: str, path: str) -> dict[str, Any]:
        """Return a keyword that must be an object, empty when absent."""
        found = node.get(keyword, {})
        if not isinstance(found, dict):
            raise ParameterSchemaError(f"{keyword} at {path} must be an object")
        return found

    def _reference(self, node: dict[str, Any], owner: str) -> None:
        """Check a ``$ref`` is local and note it for the cycle check."""
        ref = node.get("$ref")
        if ref is not None:
            if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
                raise ParameterSchemaError("only local #/$defs/... references are allowed")
            self.refs.setdefault(owner, set()).add(ref.removeprefix("#/$defs/"))

    def _branches(self, node: dict[str, Any], path: str, depth: int, owner: str) -> None:
        """Check and walk the ``anyOf``, ``oneOf``, ``allOf`` and ``prefixItems`` branches."""
        for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            branches = node.get(keyword, [])
            if not isinstance(branches, list):
                raise ParameterSchemaError(f"{keyword} at {path} must be an array")
            if len(branches) > MAX_BRANCHES:
                raise ParameterSchemaError(f"schema exceeds the {keyword} branch limit")
            for index, child in enumerate(branches):
                self.node(child, f"{path}.{keyword}[{index}]", depth + 1, owner)

    def _single_schemas(self, node: dict[str, Any], path: str, depth: int, owner: str) -> None:
        """Check and walk the subschemas that stand alone, like ``items`` and ``not``."""
        if "items" in node:
            if not isinstance(node["items"], dict):
                raise ParameterSchemaError(f"items at {path} must be an object schema")
            self.node(node["items"], f"{path}.items", depth + 1, owner)
        additional = node.get("additionalProperties")
        if isinstance(additional, dict):
            self.node(additional, f"{path}.additionalProperties", depth + 1, owner)
        elif additional is not None and not isinstance(additional, bool):
            raise ParameterSchemaError(
                f"additionalProperties at {path} must be a boolean or object schema"
            )
        for keyword in ("not", "contains", "propertyNames"):
            single = node.get(keyword)
            if isinstance(single, dict):
                self.node(single, f"{path}.{keyword}", depth + 1, owner)
            elif single is not None:
                raise ParameterSchemaError(f"{keyword} at {path} must be an object schema")

    def check_refs(self, definitions: Collection[str]) -> None:
        """Refuse a ref to a missing definition, and any cycle between definitions."""
        for targets in self.refs.values():
            if missing := targets - set(definitions):
                raise ParameterSchemaError(
                    f"unknown local schema reference: {min(missing)}. Define it under $defs"
                )
        done: set[str] = set()

        def visit(name: str, visiting: set[str]) -> None:
            """Walk the refs from one definition depth first."""
            if name in visiting:
                raise ParameterSchemaError(
                    "recursive local schema references are not supported. Inline the "
                    "definition that refers to itself"
                )
            if name in done:
                return
            visiting.add(name)
            for target in self.refs.get(name, set()):
                visit(target, visiting)
            visiting.remove(name)
            done.add(name)

        for name in definitions:
            visit(name, set())


def validate_parameter_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Check a schema against the subset.

    Args:
        schema: A JSON Schema.

    Returns:
        The same schema.

    Raises:
        ParameterSchemaError: If it uses a keyword outside the subset, breaks a limit, or is
            not a valid Draft 2020-12 schema.
    """
    if not isinstance(schema, dict):
        raise ParameterSchemaError("parameter schema must be an object")
    try:
        encoded = json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ParameterSchemaError("parameter schema must contain finite JSON values") from error
    if len(encoded.encode()) > MAX_SCHEMA_BYTES:
        raise ParameterSchemaError(f"parameter schema exceeds the {MAX_SCHEMA_BYTES}-byte limit")
    walk = _Walk()
    walk.node(schema, "$", 0, "__root__")
    walk.check_refs(schema.get("$defs", {}))
    _check_draft(encoded)
    return schema


@functools.lru_cache(maxsize=256)
def _check_draft(encoded: str) -> None:
    """Check a schema, given as its canonical JSON text, against Draft 2020-12.

    The result is cached on the text, so a schema used by many tools is checked once. A
    schema that fails is checked again each time.

    Args:
        encoded: The schema as ``json.dumps(..., sort_keys=True, separators=(",", ":"))``.

    Raises:
        ParameterSchemaError: If it is not a valid Draft 2020-12 schema.
    """
    try:
        Draft202012Validator.check_schema(json.loads(encoded))
    except SchemaError as error:
        raise ParameterSchemaError(f"invalid Draft 2020-12 schema: {error.message}") from error


def check_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Check a tool's parameter schema: the subset, with a plain object at the root.

    The root allows only ``$schema``, ``$defs``, ``type``, ``properties``, ``required``,
    ``additionalProperties``, notes and ``x-`` keys. Frameworks send just the root
    ``properties`` and ``required``, so any other root keyword would be lost on one of them.

    Args:
        schema: The JSON Schema of the tool's arguments.

    Returns:
        The same schema.

    Raises:
        ParameterSchemaError: If the schema is outside the subset or its root is not an object.
    """
    if isinstance(schema, dict) and (
        stray := {key for key in schema if not key.startswith("x-")} - ROOT_KEYWORDS
    ):
        raise ParameterSchemaError(
            f"{min(stray)} at the root of the parameters is not supported. The root is a plain"
            " object schema, because only its properties and required reach every framework."
            " Move it into a property, or wrap the choice in one property."
        )
    validate_parameter_schema(schema)
    if schema.get("type") != "object":
        raise ParameterSchemaError(
            'parameters must have "type": "object" at the root, with the arguments as properties'
        )
    return schema


def model_schema(parameters: dict[str, Any], hidden: Collection[str]) -> dict[str, Any]:
    """Return the parameters without the hidden arguments, as the model sees them."""
    if not hidden:
        return parameters
    shown = dict(parameters)
    if "properties" in shown:
        shown["properties"] = {
            name: value for name, value in shown["properties"].items() if name not in hidden
        }
    if "required" in shown:
        shown["required"] = [name for name in shown["required"] if name not in hidden]
    return shown


FORMAT_WORDS = MappingProxyType(
    {
        "birth-date": "a date of birth written YYYY-MM-DD",
        "date": "a date written YYYY-MM-DD",
        "date-time": "a date and time written YYYY-MM-DDTHH:MM",
        "email": "an email address",
        "id": "an identifier: letters, digits, and then any of dot, dash, underscore or colon",
        "name-email": "a name and an email address, written Jo Bloggs <jo@example.com>",
        "phone": "a phone number with its country code: a plus, the country code, then the number",
        "spelled": "the code as one word of letters and digits, like KP482913",
        "time": "a time of day written HH:MM on the 24-hour clock",
    }
)
"""What each ``format`` the library knows asks for, in the words the model reads."""
_LOCAL_REF = "#/$defs/"
_LITERAL_KEYWORDS = frozenset({"enum", "const", "default", "examples", "example"})
_SCHEMA_MAPS = frozenset({"properties", "patternProperties"})
_DROPPED_KEYWORDS = frozenset({"format", "pattern", "$defs", "title"})
_NULL = MappingProxyType({"type": "null"})


def model_shape(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a tool's parameters as the model sees them: plain, with nothing it must decode.

    - Every local ``$ref`` is inlined, and ``$defs`` and ``title`` go: a ref the model cannot
      resolve makes it invent field names.
    - ``format`` and ``pattern`` become words in the description: some providers refuse
      them, and a format only this library knows, like ``spelled``, tells a model nothing.
    - A property its object does not require, whose ``default`` is null, loses its null
      branch and that default: leaving it out already says "no value". Any other property
      keeps null, which is then a real answer, like "clear this value".

    The arguments are still checked against the full schema, so a model that sends null
    for an optional value is accepted.

    Args:
        schema: The parameters, a schema the library accepts.
    """
    shaped = _shape(schema, schema.get("$defs", {}))
    assert isinstance(shaped, dict)  # ponytail: a dict shapes to a dict, for the type checker
    return shaped


def _shape(node: object, defs: dict[str, Any]) -> object:
    """Shape one schema node. See ``model_shape``."""
    if isinstance(node, list):
        return [_shape(item, defs) for item in node]
    if not isinstance(node, dict):
        return node
    target = node.get("$ref")
    if isinstance(target, str) and target.startswith(_LOCAL_REF):
        found = _shape(defs.get(target.removeprefix(_LOCAL_REF), {}), defs)
        siblings = {key: _keyword(key, value, defs) for key, value in node.items() if key != "$ref"}
        return {**found, **siblings} if isinstance(found, dict) else found
    shaped = {
        key: _keyword(key, value, defs)
        for key, value in node.items()
        if key not in _DROPPED_KEYWORDS
    }
    if words := _format_words(node):
        shaped["description"] = _joined(shaped.get("description"), words)
    if isinstance(properties := shaped.get("properties"), dict):
        listed = shaped.get("required")
        required = set(listed) if isinstance(listed, list) else set()
        shaped["properties"] = {
            name: _optional(child) if name not in required and _defaults_to_null(child) else child
            for name, child in properties.items()
        }
    return shaped


def _keyword(keyword: str, value: object, defs: dict[str, Any]) -> object:
    """Shape the value of one keyword: literals stay, a property map keeps its names."""
    if keyword in _LITERAL_KEYWORDS:
        return value
    if keyword in _SCHEMA_MAPS and isinstance(value, dict):
        return {name: _shape(child, defs) for name, child in value.items()}
    return _shape(value, defs)


def _format_words(node: dict[str, Any]) -> str:
    """Return what a node's ``format`` and ``pattern`` ask for as a sentence, or ""."""
    words = []
    if node.get("format") in FORMAT_WORDS:
        words.append(FORMAT_WORDS[node["format"]])
    if isinstance(node.get("pattern"), str):
        words.append("text matching " + node["pattern"])
    return f"Expected {'; '.join(words)}." if words else ""


def _defaults_to_null(node: object) -> bool:
    """Return True when a property's ``default`` is null: sending null equals leaving it out."""
    return isinstance(node, dict) and "default" in node and node["default"] is None


def _optional(node: dict[str, Any]) -> dict[str, Any]:
    """Return an optional property without its null branch and its ``default: null``."""
    shaped = {key: value for key, value in node.items() if key != "default"}
    if isinstance(types := shaped.get("type"), list) and "null" in types:
        left = [kind for kind in types if kind != "null"]
        shaped["type"] = left[0] if len(left) == 1 else left
    branches = shaped.get("anyOf")
    if not isinstance(branches, list) or _NULL not in branches:
        return shaped
    kept = [branch for branch in branches if branch != _NULL]
    if len(kept) != 1 or not isinstance(kept[0], dict):
        return {**shaped, "anyOf": kept} if kept else shaped
    rest = {key: value for key, value in shaped.items() if key != "anyOf"}
    description = _joined(rest.get("description"), kept[0].get("description"))
    merged = {**kept[0], **rest}
    return {**merged, "description": description} if description else merged


def _joined(first: object, second: object) -> str:
    """Return two descriptions as one text, either one left out when empty."""
    return " ".join(str(part) for part in (first, second) if part)
