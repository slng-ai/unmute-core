"""The contract check itself, against the Unmute CLI's own sample lines and bad records."""

import json
from pathlib import Path

import pytest

from tests.dev.contract import problems

DATA = Path(__file__).parent.parent / "data" / "devmetrics"
SAMPLES = [
    (path.name, line)
    for path in sorted(DATA.glob("*.jsonl"))
    for line in path.read_text().splitlines()
]
VALID = (
    '{"version":2,"kind":"state","call_id":"call-a","id":"state-1","revision":1,"order":1,'
    '"state":{"origin":{"kind":"step","name":"booking"},"changes":[{"field":"appointment",'
    '"value":{"time":"16:00:00"},"confirmed":true,"by":{"kind":"step","name":"booking"}}]}}'
)


@pytest.mark.parametrize(("name", "line"), SAMPLES)
def test_every_sample_line_of_the_unmute_cli_passes(name: str, line: str) -> None:
    assert problems(line.removeprefix("UNMUTE_METRIC ")) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"kind":"step","name":"booking"},"changes"', '"kind":"guess","name":"booking"},"changes"'),
        ('"by":{"kind":"step"', '"by":{"kind":"magic"'),
        ('"field":"appointment"', '"field":""'),
        ('"field":"appointment"', '"field":"two words"'),
        ('"value":{"time":"16:00:00"},', ""),
        ('"confirmed":true', '"confirmed":true,"prompt":"private"'),
        ('"version":2', '"version":3'),
        ('"revision":1', '"revision":0'),
    ],
)
def test_the_check_refuses_what_the_go_reader_refuses(old: str, new: str) -> None:
    assert problems(VALID.replace(old, new, 1))


def test_a_valid_state_line_passes() -> None:
    assert problems(VALID) == []
    assert json.loads(VALID)["kind"] == "state"
