"""Reasoning tags never reach the caller, split across chunks or not."""

import pytest

from unmute_core.model import MODEL_REASONING_MARKERS, ReasoningFilter, reasoning_markers

GEMMA = ("<|channel>thought", "<channel|>")


def _run(*chunks: str, markers: tuple[tuple[str, str], ...] = ()) -> str:
    speech = ReasoningFilter(markers)
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


@pytest.mark.parametrize(
    ("chunks", "spoken"),
    [
        (("<|channel>thought\nI should check.<channel|>Is that right?",), "Is that right?"),
        (("<|channel>thought\n<channel|>Yes.",), "Yes."),
        (
            (
                "<|chan",
                "nel>thou",
                "ght\nweigh",
                "ing<chan",
                "nel|",
                ">Hi there",
            ),
            "Hi there",
        ),
        (("Sure.<|channel>thought x<channel|> Done.",), "Sure. Done."),
        (("<|channel>thought still weighing", " the options"), ""),
        (("Hello. <|channel>thought weighing",), "Hello. "),
        (("<channel|>Okay.",), "Okay."),
        (("<think>a</think>b<|channel>thought c<channel|>d",), "bd"),
        (("<|channel>thought a </think> b <think> c<channel|>d",), "d"),
        (("<think>a <channel|> b</think>c",), "c"),
        (("a < b and <|c and 3<4",), "a < b and <|c and 3<4"),
        (("<channel>x</channel>",), "<channel>x</channel>"),
        (("<|CHANNEL>thought x <CHANNEL|>y",), "<|CHANNEL>thought x <CHANNEL|>y"),
        (("ends with <|chan",), "ends with <|chan"),
        (("ends with <",), "ends with <"),
    ],
)
def test_a_models_own_markers_never_reach_the_caller(chunks, spoken):
    assert _run(*chunks, markers=(GEMMA,)) == spoken


def test_markers_are_matched_as_written_not_as_a_pattern():
    assert _run("a.b<x>c", "<y>", markers=(("a.b", "<x>"),)) == "c<y>"
    assert _run("axb and c", markers=(("a.b", "<x>"),)) == "axb and c"


def test_several_pairs_each_close_only_their_own_block():
    pairs = (GEMMA, ("[[", "]]"))
    assert _run("[[a<channel|>b]]c", markers=pairs) == "c"
    assert _run("<|channel>thought a]]b<channel|>c", markers=pairs) == "c"


def test_the_standard_tags_are_unchanged_when_markers_are_given():
    for chunks in (
        ("<th", "ink>che", "ck</thi", "nk>Is that right?"),
        ("Sure.</", "think>", " Done."),
    ):
        assert _run(*chunks, markers=(GEMMA,)) == _run(*chunks)


@pytest.mark.parametrize(
    ("model", "markers"),
    [
        ("google/gemma-4-31b-it", (GEMMA,)),
        ("gemma-4-31b-it", ()),
        ("vertex_ai/google/gemma-4-31b-it", ()),
        ("Google/Gemma-4-31B-it", ()),
        ("gpt-4o", ()),
        ("openai/gpt-4o", ()),
        ("gemma-4-31b-it-extra", ()),
        ("unknown", ()),
        ("", ()),
    ],
)
def test_a_model_names_its_markers_by_its_exact_id(model, markers):
    assert reasoning_markers(model) == markers


def test_the_per_model_markers_are_the_ones_livekits_own_models_strip():
    assert MODEL_REASONING_MARKERS == {"google/gemma-4-31b-it": GEMMA}


@pytest.mark.parametrize(
    "pair", [("", "</x>"), ("<x>", ""), ("<x>", "<x>")], ids=["no open", "no close", "same"]
)
def test_a_marker_pair_that_cannot_open_and_close_is_refused(pair):
    with pytest.raises(ValueError, match="two different, non-empty markers"):
        ReasoningFilter([pair])
