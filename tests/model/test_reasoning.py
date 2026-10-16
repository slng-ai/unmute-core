"""Reasoning tags never reach the caller, split across chunks or not."""

import pytest

from unmute_core.model import ReasoningFilter


def _run(*chunks: str) -> str:
    speech = ReasoningFilter()
    return "".join(speech.feed(chunk) for chunk in chunks) + speech.flush()


@pytest.mark.parametrize(
    ("chunks", "spoken"),
    [
        (("</think>Okay, so I've got your details.",), "Okay, so I've got your details."),
        (
            ("I'll just get you on the books now.\n</think>\n",),
            "I'll just get you on the books now.\n\n",
        ),
        (("<think>check the number</think>Is that right?",), "Is that right?"),
        (("<th", "ink>che", "ck</thi", "nk>Is that right?"), "Is that right?"),
        (("Sure.</", "think>", " Done."), "Sure. Done."),
        (("<thinking>a</think>b</thinking>Hi",), "Hi"),
        (("<reasoning>x</reasoning>Yes",), "Yes"),
        (("Hello. <think>weighing", " options"), "Hello. "),
        (("a < b and 3<4",), "a < b and 3<4"),
        (("see <thinker> and <b>bold</b>",), "see <thinker> and <b>bold</b>"),
        (("ends with <",), "ends with <"),
        (("ends with <thi",), "ends with <thi"),
    ],
)
def test_reasoning_tags_never_reach_the_caller(chunks, spoken):
    assert _run(*chunks) == spoken
