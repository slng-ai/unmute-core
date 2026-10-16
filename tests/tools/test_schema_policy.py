"""The parameter schema subset: the keywords it accepts and refuses, and its size limits."""

import pytest
from pydantic import BaseModel

from unmute_core.tools import ParameterSchemaError, check_parameters
from unmute_core.tools._schema import model_schema, validate_parameter_schema


@pytest.mark.parametrize(
    "schema",
    [
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"Id": {"type": "string", "minLength": 1}},
            "type": "object",
            "properties": {"id": {"$ref": "#/$defs/Id"}},
        },
        {"type": "object", "properties": {"day": {"type": "string", "format": "date"}}},
        {"type": "object", "additionalProperties": False},
        {"type": "object", "properties": {"a": {"type": "string", "pattern": "^[a-z]+$"}}},
        {"type": "object", "$comment": "note", "x-foo": {"any": 1}},
        {"type": "object", "properties": {"a": {"discriminator": {}}}},
        {"type": "object", "properties": {"a": {"type": "string", "example": "x"}}},
        {"type": "object", "properties": {"m": {"minProperties": 1, "maxProperties": 3}}},
        {"type": "object", "properties": {"m": {"patternProperties": {"^x": {"type": "string"}}}}},
        {"type": "object", "properties": {"m": {"propertyNames": {"pattern": "^[a-z]+$"}}}},
        {"type": "object", "properties": {"a": {"type": "array", "contains": {"type": "integer"}}}},
        {
            "type": "object",
            "properties": {"a": {"contains": {}, "minContains": 1, "maxContains": 2}},
        },
        {
            "type": "object",
            "properties": {
                "a": {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}]}
            },
        },
        {"type": "object", "properties": {"n": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}},
    ],
)
def test_accepted(schema):
    assert check_parameters(schema) is schema


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"$ref": "https://example.com/schema.json"}, "only local"),
        ({"$dynamicRef": "#node"}, "unsupported schema keyword"),
        ({"$defs": {"Node": {"$ref": "#/$defs/Node"}}, "$ref": "#/$defs/Node"}, "recursive local"),
        ({"$ref": "#/$defs/Missing"}, "unknown local schema reference: Missing"),
        ({"type": "not-a-json-schema-type"}, "invalid Draft 2020-12"),
        ({"definitions": {}}, "Use \\$defs"),
        ({"$id": "https://example.com/x"}, "unsupported schema keyword"),
        ({"contains": True}, "contains .* object schema"),
        ({"$schema": "https://json-schema.org/draft/2019-09/schema"}, "must be exactly"),
        ({"type": "array", "items": True}, "items .* object schema"),
        ({"not": False}, "not .* object schema"),
        ({"type": "number", "const": float("inf")}, "finite JSON"),
        ({"type": "number", "minimum": float("inf")}, "finite JSON"),
        ({"type": "string", "enum": list(range(257))}, "256-item limit"),
        ({"type": "object", "required": [str(i) for i in range(257)]}, "256-item limit"),
        ({"type": "object", "required": ("a",)}, "must be an array"),
        ({"type": "object", "properties": {"a": {"description": "x" * 70_000}}}, "byte limit"),
        ("not a schema", "must be an object"),
        (
            {"type": "object", "properties": {"a": {"type": "string", "nullable": True}}},
            r'Write "type": \["<type>", "null"\] instead',
        ),
    ],
)
def test_refused(schema, message):
    with pytest.raises(ParameterSchemaError, match=message):
        validate_parameter_schema(schema)


def test_deep_schema_is_refused():
    schema: dict = {"type": "string"}
    for _ in range(40):
        schema = {"type": "array", "items": schema}
    with pytest.raises(ParameterSchemaError, match="depth limit"):
        validate_parameter_schema(schema)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("patternProperties", {"^x": {"type": "string"}}),
        ("oneOf", [{"required": ["a"]}]),
        ("anyOf", [{"required": ["a"]}]),
        ("allOf", [{"required": ["a"]}]),
        ("not", {"required": ["b"]}),
        ("propertyNames", {"pattern": "^[a-z]+$"}),
        ("minProperties", 1),
        ("maxProperties", 3),
        ("if", {}),
    ],
)
def test_the_root_of_the_parameters_is_a_plain_object_schema(keyword, value):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, keyword: value}
    with pytest.raises(ParameterSchemaError, match=f"{keyword} at the root.*wrap the choice"):
        check_parameters(schema)


def test_a_pydantic_root_still_passes():
    class Book(BaseModel):
        title: str
        tags: list[str] = []

    schema = Book.model_json_schema()
    assert check_parameters(schema) is schema


def test_the_root_keeps_notes_and_extension_keys():
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {},
        "type": "object",
        "title": "T",
        "description": "d",
        "$comment": "c",
        "additionalProperties": False,
        "x-note": 1,
    }
    assert check_parameters(schema) is schema


@pytest.mark.parametrize("schema", [{"type": "string"}, {"properties": {}}])
def test_tool_parameters_need_an_object_root(schema):
    with pytest.raises(ParameterSchemaError, match='"type": "object"'):
        check_parameters(schema)


def test_model_schema_drops_hidden_arguments_and_keeps_the_original():
    parameters = {
        "type": "object",
        "properties": {"day": {"type": "string"}, "phone": {"type": "string"}},
        "required": ["day", "phone"],
    }
    shown = model_schema(parameters, {"phone"})
    assert shown == {
        "type": "object",
        "properties": {"day": {"type": "string"}},
        "required": ["day"],
    }
    assert "phone" in parameters["properties"]


def test_a_repeated_schema_is_checked_against_draft_2020_12_once():
    from unmute_core.tools._schema import _check_draft

    schema = {"type": "object", "properties": {"repeat_once": {"type": "string"}}}
    # one cache miss per schema keeps the Draft 2020-12 check from running every turn
    before = _check_draft.cache_info()
    for _ in range(3):
        validate_parameter_schema(schema)
    after = _check_draft.cache_info()
    assert after.misses - before.misses == 1
    assert after.hits - before.hits == 2
