"""A State built from declared variables and shapes, each typed with type text.

The cases cover values, shapes and the refusals of ``StateSchema`` and ``CallState``.
"""

import datetime
import gc
import json
import weakref
from typing import get_args

import pytest
from pydantic import ValidationError

from unmute_core.state import CallState, StateRefused, StateSchema, StateSchemaError
from unmute_core.tasks import TaskSpec
from unmute_core.tasks._values import finish_parameters, result_model, save_task_values

SHAPES = [
    {
        "name": "Appointment",
        "description": "One booking.",
        "fields": [
            {"name": "day", "type": "Date", "description": "The day booked."},
            {"name": "note", "type": "str | None"},
        ],
    },
    {
        "name": "Customer",
        "fields": [
            {"name": "contact", "type": "NameEmail", "description": "Who."},
            {"name": "status", "type": 'Literal["new", "known"]'},
        ],
    },
]
VARIABLES = [
    {"name": "customer", "description": "The caller", "type": "Customer | None"},
    {"name": "bookings", "description": "Held", "type": "list[Appointment]"},
    {"name": "order_id", "description": "The order", "type": "Id", "confirm": "find_order"},
    {"name": "note", "description": "A note"},
]


def _schema() -> StateSchema:
    return StateSchema(VARIABLES, SHAPES)


def test_every_value_starts_empty_and_confirm_comes_from_the_rows():
    schema = _schema()
    state = CallState(schema.model, confirm=schema.confirm)
    assert state.plain("customer") is None
    assert state.plain("bookings") == []
    assert schema.confirm == {"order_id": "find_order"}
    assert schema.types["bookings"].canonical == "list[Appointment]"
    assert schema.model.model_fields["customer"].description == "The caller"


def test_a_shape_is_checked_field_by_field():
    state = CallState(_schema().model)
    state.save({"customer": {"contact": "Fred Bloggs <fred@Example.com>", "status": "new"}})
    assert state.plain("customer") == {
        "contact": {"name": "Fred Bloggs", "email": "fred@example.com"},
        "status": "new",
    }
    assert state.plain("customer.contact.email") == "fred@example.com"
    with pytest.raises(StateRefused, match=r"^customer\.status: Input should be 'new' or"):
        state.save({"customer": {"contact": "fred@example.com", "status": "vip"}})
    with pytest.raises(StateRefused, match="expected an email address"):
        state.save({"customer": {"contact": "not an address", "status": "new"}})


def test_a_list_checks_every_entry():
    state = CallState(_schema().model)
    state.save({"bookings": [{"day": "2026-03-19"}]})
    assert state.plain("bookings") == [{"day": "2026-03-19", "note": None}]
    with pytest.raises(StateRefused, match=r"^bookings\.0\.day: .*expected a day"):
        state.save({"bookings": [{"day": "tomorrow"}]})


def test_empty_text_is_no_value_yet_and_a_number_is_text():
    state = CallState(_schema().model)
    state.save({"customer": "", "note": 42})
    assert state.plain("customer") is None
    assert state.plain("note") == "42"
    state.save({"order_id": 1234})
    assert state.plain("order_id") == "1234"
    with pytest.raises(StateRefused):
        state.save({"note": True})


def test_a_date_inside_a_shape_reads_back_as_its_text():
    state = CallState(_schema().model)
    state.save({"bookings": [{"day": datetime.date(2026, 3, 19), "note": "x"}]})
    assert state.render("bookings") == '[{"day":"2026-03-19","note":"x"}]'


def test_a_shape_field_may_start_with_model_when_pydantic_does_not_use_it():
    schema = StateSchema(
        [{"name": "car", "type": "Car | None"}],
        [{"name": "Car", "fields": [{"name": "model_year", "type": "int"}]}],
    )
    state = CallState(schema.model)
    state.save({"car": {"model_year": 2020}})
    assert state.plain("car") == {"model_year": 2020}


def test_a_finish_describes_the_formats_inside_a_shape():
    schema = _schema()
    task = TaskSpec(
        name="book",
        when="The caller books.",
        instructions="Book.",
        assign=[
            {"variable": "customer", "path": ["customer"]},
            {"variable": "bookings", "path": ["booking"], "append": True},
        ],
    )
    parameters = finish_parameters(result_model(task, schema.model), record=True)
    day = parameters["properties"]["booking"]["anyOf"][0]["properties"]["day"]
    assert day["description"] == "The day booked. Expected a date written YYYY-MM-DD."
    assert "$ref" not in json.dumps(parameters)


def test_appending_a_name_email_with_no_address_adds_nothing():
    schema = StateSchema([{"name": "guests", "type": "list[NameEmail]"}])
    state = CallState(schema.model)
    task = TaskSpec(
        name="book",
        when="The caller books.",
        instructions="Book.",
        assign=[{"variable": "guests", "path": ["guest"], "append": True}],
    )
    model = result_model(task, schema.model)
    for guest in ("ana@example.com", ""):
        save_task_values(task, {"guest": guest}, state=state, model=model)
    assert state.plain("guests") == [{"name": "ana", "email": "ana@example.com"}]


@pytest.mark.parametrize(
    ("variables", "shapes", "message"),
    [
        (
            [{"name": "c", "type": "Client"}],
            SHAPES,
            "runtime variable 'c': no shape named 'Client'",
        ),
        (
            VARIABLES,
            [
                SHAPES[0],
                {**SHAPES[1], "fields": [*SHAPES[1]["fields"], {"name": "me", "type": "Customer"}]},
            ],
            "shape 'Customer' refers to itself, through Customer -> Customer",
        ),
        (
            [],
            [
                {"name": "A", "fields": [{"name": "b", "type": "B | None"}]},
                {"name": "B", "fields": [{"name": "a", "type": "list[A]"}]},
            ],
            "through A -> B -> A",
        ),
        ([], [SHAPES[0], SHAPES[0]], "shape 'Appointment' is declared twice"),
        ([{"name": "x"}, {"name": "x"}], [], "runtime variable 'x' is declared twice"),
        (
            [],
            [{"name": "A", "fields": [{"name": "b", "type": "Missing"}]}],
            "shape 'A' field 'b': no shape named 'Missing'",
        ),
    ],
)
def test_rows_that_do_not_fit_together_are_refused(variables, shapes, message):
    with pytest.raises(StateSchemaError, match=message):
        StateSchema(variables, shapes)


@pytest.mark.parametrize(
    ("variables", "shapes", "message"),
    [
        ([], [{**SHAPES[0], "name": "Phone"}], "cannot be called that"),
        (
            [],
            [{"name": "A", "fields": [{"name": "model_dump", "type": "str"}]}],
            "cannot be called 'model_dump': Pydantic",
        ),
        (
            [],
            [{"name": "A", "fields": [{"name": "model_config", "type": "str"}]}],
            "cannot be called 'model_config': Pydantic",
        ),
        ([], [{"name": "A", "fields": [{"name": "Bad", "type": "str"}]}], "lower case"),
        (
            [{"name": "x", "type": "list[" * 20 + "str" + "]" * 20}],
            [],
            "nests more than 16 brackets deep",
        ),
        ([{"name": "x", "type": "datetime"}], [], 'write "Date" and "Time"'),
        ([{"name": "1x"}], [], "String should match pattern"),
    ],
)
def test_malformed_rows_are_refused(variables, shapes, message):
    with pytest.raises(ValidationError, match=message):
        StateSchema(variables, shapes)


def test_a_cycle_is_named_from_the_first_shape_in_name_order() -> None:
    shapes = [
        {"name": "A", "fields": [{"name": "x", "type": "str"}]},
        {"name": "B", "fields": [{"name": "d", "type": "D | None"}]},
        {"name": "C", "fields": [{"name": "d", "type": "D | None"}]},
        {
            "name": "D",
            "fields": [{"name": "a", "type": "A | None"}, {"name": "b", "type": "B | None"}],
        },
    ]
    with pytest.raises(
        StateSchemaError, match=r"shape 'B' refers to itself, through B -> D -> B\."
    ):
        StateSchema([], shapes)


def test_schemas_built_per_call_do_not_pile_up_in_a_cache() -> None:
    shapes = [
        {"name": "Customer", "fields": [{"name": "status", "type": 'Literal["new", "known"]'}]}
    ]
    variables = [{"name": "customer", "description": "Who calls.", "type": "Customer | None"}]
    task = TaskSpec(
        name="t",
        when="w",
        instructions="i",
        assign=[{"variable": "customer", "path": ["customer"]}],
    )
    kept = []
    for _ in range(300):
        schema = StateSchema(variables, shapes)
        optional = schema.model.model_fields["customer"].annotation
        kept.append(weakref.ref(get_args(get_args(optional)[0])[0]))
        result_model(task, schema.model)
        del schema
    gc.collect()
    assert sum(ref() is not None for ref in kept) < 200


NAMES_PYDANTIC_CANNOT_TAKE = ["_1st_choice", "_x", "model_dump", "model_config"]


def _awkward_schema() -> StateSchema:
    return StateSchema([{"name": name, "type": "str"} for name in NAMES_PYDANTIC_CANNOT_TAKE])


def test_a_name_that_would_share_a_field_with_an_aliased_name_is_refused():
    with pytest.raises(StateSchemaError, match=r"\['_x', 'v___x'\] would share the field 'v___x'"):
        StateSchema([{"name": "_x"}, {"name": "v___x"}])


def test_names_pydantic_cannot_take_save_render_and_snapshot_under_their_real_names():
    state = CallState(_awkward_schema().model)
    state.save({name: f"{name} value" for name in NAMES_PYDANTIC_CANNOT_TAKE})
    for name in NAMES_PYDANTIC_CANNOT_TAKE:
        assert state.plain(name) == f"{name} value"
        assert state.render(name) == f"{name} value"
    assert state.snapshot() == {name: f"{name} value" for name in NAMES_PYDANTIC_CANNOT_TAKE}
    assert state.values.model_dump(by_alias=True)["_x"] == "_x value"


def test_a_finish_for_names_pydantic_cannot_take_shows_and_saves_the_real_names():
    schema = _awkward_schema()
    task = TaskSpec(
        name="pick",
        when="The caller picks.",
        instructions="Pick.",
        assign=[{"variable": name, "path": [name]} for name in NAMES_PYDANTIC_CANNOT_TAKE],
    )
    model = result_model(task, schema.model)
    properties = finish_parameters(model, record=False)["properties"]
    assert set(NAMES_PYDANTIC_CANNOT_TAKE) <= set(properties)
    assert not any(name.startswith("v__") for name in properties)
    state = CallState(schema.model)
    arguments = dict.fromkeys(NAMES_PYDANTIC_CANNOT_TAKE, "chosen")
    save_task_values(task, arguments, state=state, model=model)
    assert state.plain("_x") == "chosen"
    assert state.plain("model_dump") == "chosen"


async def test_the_values_tool_names_and_saves_values_pydantic_cannot_take():
    from unmute_core.tools import values_block, values_tool

    state = CallState(_awkward_schema().model)
    tool = values_tool(state)
    assert set(NAMES_PYDANTIC_CANNOT_TAKE) <= set(tool.parameters["properties"])
    await tool.executor.run({"_x": "hi", "model_config": "cfg"}, state)
    assert state.plain("_x") == "hi"
    block = values_block(state)
    assert "_x" in block and "model_config" in block
