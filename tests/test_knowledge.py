"""Knowledge: a small base built, baked, loaded and queried, and every startup failure loud.

Ported from unmute@d0c67f1 internal/generate/knowledge_smoke_test.go and
knowledge_startup_test.go. The embedder is llama-index's MockEmbedding, so no network.
"""

import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from unmute_core.knowledge import KnowledgeBase, KnowledgeError
from unmute_core.knowledge._base import _clean, _Hit, _merge

POLICY = (
    "Refunds are issued within fourteen days of the visit.\n\n"
    "To ask for one, quote the refund code RC-2026-04 when you call the salon.\n\n"
    "Gift cards cannot be refunded, only exchanged for another treatment."
)
HOURS = "The salon opens at nine in the morning and closes at seven in the evening."


class CorpusNeverEmbedded(MockEmbedding):
    """Embeds questions only. Fails if anything tries to embed the documents."""

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the corpus was embedded")


class FakeRetriever:
    """Returns fixed hits, or raises."""

    def __init__(self, hits: list[NodeWithScore] | Exception) -> None:
        self.hits = hits

    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        if isinstance(self.hits, Exception):
            raise self.hits
        return self.hits


def mock_embed() -> MockEmbedding:
    """A tiny embedder that needs no network."""
    return MockEmbedding(embed_dim=8)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A refunds base with two documents, inside a hidden folder like a checkout path."""
    folder = tmp_path / ".checkout" / "refunds"
    folder.mkdir(parents=True)
    (folder / "policy.md").write_text(POLICY)
    (folder / "hours.txt").write_text(HOURS)
    return folder


def base(folder: Path, **settings: Any) -> KnowledgeBase:
    """A hybrid base on the folder unless the settings say otherwise."""
    if settings.get("mode", "hybrid") != "keyword":
        settings.setdefault("embed", mock_embed)
    return KnowledgeBase("refunds", folder, **settings)


async def test_a_hybrid_base_answers_with_passages_and_bare_file_names(folder: Path) -> None:
    refunds = base(folder, chunk_size=40, chunk_overlap=0)
    refunds.start()

    found = await refunds.look_up("refund code RC-2026-04")

    results = found["results"]
    assert 1 <= len(results) <= 3
    assert any("RC-2026-04" in result["text"] for result in results)
    assert {result["source"] for result in results} <= {"policy.md", "hours.txt"}
    assert any("score" in result for result in results)


async def test_keyword_search_ranks_the_verbatim_code_first_and_gives_no_score(
    folder: Path,
) -> None:
    refunds = base(folder, mode="keyword", chunk_size=40, chunk_overlap=0)
    refunds.start()

    results = (await refunds.look_up("rc-2026-04"))["results"]

    assert "RC-2026-04" in results[0]["text"]
    assert all("score" not in result for result in results)


async def test_a_passage_keeps_only_its_file_name_and_hides_it_from_the_model(
    folder: Path,
) -> None:
    refunds = base(folder, mode="keyword")
    refunds.start()
    assert refunds._keyword is not None

    node = (await refunds._keyword.aretrieve("refund"))[0].node

    assert node.metadata == {"file_name": "policy.md"}
    assert node.excluded_embed_metadata_keys == ["file_name"]
    assert node.excluded_llm_metadata_keys == ["file_name"]


def test_the_halves_are_interleaved_and_a_repeat_keeps_its_score() -> None:
    meaning = [
        _Hit(id="d1", text="a", source="x.md", score=0.9),
        _Hit(id="d2", text="b", source="x.md", score=0.8),
        _Hit(id="d3", text="c", source="x.md", score=0.7),
    ]
    keyword = [
        _Hit(id="e1", text="d", source="y.md"),
        _Hit(id="d1", text="a", source="x.md"),
        _Hit(id="e2", text="e", source="y.md"),
    ]

    assert [hit.id for hit in _merge(meaning, keyword, 3)] == ["d1", "e1", "d2"]
    assert _merge(meaning, keyword, 5)[0] == _Hit(id="d1", text="a", source="x.md", score=0.9)
    assert [hit.id for hit in _merge([], keyword, 2)] == ["e1", "d1"]


async def test_min_score_drops_only_scored_hits(
    folder: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, "unmute_core.knowledge")
    refunds = base(folder, min_score=0.5)
    refunds.start()
    weak = NodeWithScore(node=TextNode(id_="weak", text="weak"), score=0.2)
    strong = NodeWithScore(node=TextNode(id_="strong", text="strong"), score=0.9)
    word = NodeWithScore(node=TextNode(id_="word", text="word"), score=12.0)
    refunds._vector = FakeRetriever([strong, weak])  # ty: ignore[invalid-assignment]
    refunds._keyword = FakeRetriever([word])  # ty: ignore[invalid-assignment]

    results = (await refunds.look_up("anything"))["results"]

    assert [result["text"] for result in results] == ["strong", "word"]
    assert results[0]["score"] == 0.9
    assert "knowledge 'refunds': dropped 1 result(s) below min_score 0.5" in caplog.messages


async def test_a_baked_base_loads_without_embedding_the_corpus(
    folder: Path, tmp_path: Path
) -> None:
    index_dir = tmp_path / "knowledge_index" / "refunds"
    base(folder, index_dir=index_dir).bake()
    assert (index_dir / "baked.json").is_file()

    refunds = base(folder, index_dir=index_dir, embed=lambda: CorpusNeverEmbedded(embed_dim=8))
    refunds.start()

    assert (await refunds.look_up("refund code"))["results"]


def test_a_base_baked_with_other_settings_is_refused(folder: Path, tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    base(folder, index_dir=index_dir, top_k=3).bake()

    with pytest.raises(KnowledgeError, match="baked index was built with other settings"):
        base(folder, index_dir=index_dir, top_k=4).start()


def test_starting_without_a_baked_index_writes_nothing(folder: Path, tmp_path: Path) -> None:
    before = sorted(tmp_path.rglob("*"))

    base(folder, index_dir=tmp_path / "index").start()

    assert sorted(tmp_path.rglob("*")) == before


def test_bake_needs_somewhere_to_write(folder: Path) -> None:
    with pytest.raises(KnowledgeError, match="bake needs an index_dir"):
        base(folder).bake()


async def test_a_lookup_that_cannot_search_says_so_without_the_question(
    folder: Path, caplog: pytest.LogCaptureFixture
) -> None:
    refunds = base(folder, mode="keyword")
    assert await refunds.look_up("my card number is 4111") == {"error": "lookup unavailable"}

    refunds.start()
    refunds._keyword = FakeRetriever(RuntimeError("store down"))  # ty: ignore[invalid-assignment]
    assert await refunds.look_up("my card number is 4111") == {"error": "lookup unavailable"}
    assert "4111" not in caplog.text


async def test_a_lookup_logs_counts_never_the_question_or_a_passage(
    folder: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, "unmute_core.knowledge")
    refunds = base(folder, mode="keyword")
    refunds.start()

    await refunds.look_up("when do you open")

    assert any("returned" in message and "result(s) in" in message for message in caplog.messages)
    assert "when do you open" not in caplog.text
    assert "nine in the morning" not in caplog.text


def test_a_missing_credential_fails_at_start_and_names_the_variable(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EMBED_TEST_KEY", raising=False)

    with pytest.raises(KnowledgeError, match="EMBED_TEST_KEY is not set"):
        base(folder, credential_env="EMBED_TEST_KEY").start()


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (None, "is not a folder"),
        ({}, "read no documents"),
        ({"scan.txt": "%PDF-1.7 binary"}, "was read as raw PDF bytes"),
        ({"blank.txt": "   \n"}, "no document yielded any text"),
    ],
)
def test_a_base_that_would_know_nothing_fails_at_start(
    tmp_path: Path, files: dict[str, str] | None, message: str
) -> None:
    folder = tmp_path / "docs"
    if files is not None:
        folder.mkdir()
        for name, text in files.items():
            (folder / name).write_text(text)

    with pytest.raises(KnowledgeError, match=message):
        base(folder, mode="keyword").start()


def test_a_document_without_text_is_skipped_with_a_warning(
    folder: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (folder / "scanned.txt").write_text("\n\n")

    base(folder, mode="keyword").start()

    assert "knowledge 'refunds': skipping scanned.txt, no text layer" in caplog.messages


def test_layout_characters_are_removed_and_line_breaks_kept() -> None:
    assert _clean("Cut\x7f and\x00 dry\n\tnext     line\r") == "Cut and dry\n\tnext line\r"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"chunk_size": 50, "chunk_overlap": 60}, "chunk_overlap 60 must be 0 to chunk_size"),
        ({"mode": "keyword", "min_score": 0.3}, "keyword hits have no score"),
        ({"mode": "meaning", "embed": None}, "needs an embedder"),
        ({"mode": "keyword", "embed": mock_embed}, "leave out embed"),
    ],
)
def test_settings_are_checked_when_the_base_is_built(
    folder: Path, settings: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KnowledgeBase("refunds", folder, **settings)


async def test_the_lookup_tool_gives_the_model_the_schema_and_the_relevance_note(
    folder: Path,
) -> None:
    refunds = base(folder, chunk_size=40, chunk_overlap=0)
    refunds.start()

    plain = refunds.tool()
    described = refunds.tool("find_policy", "Look up refund rules.")

    assert plain.name == "look_up"
    assert plain.description.startswith("The results may not answer the question.")
    assert described.name == "find_policy"
    assert described.description.startswith("Look up refund rules.\n\nThe results may not")
    assert plain.parameters == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up. Use the caller's own words.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    query = "refund code RC-2026-04"
    assert await plain.executor.run({"query": query}, None) == await refunds.look_up(query)


def test_a_range_error_names_the_public_class_and_the_field(folder: Path) -> None:
    with pytest.raises(ValueError) as caught:
        KnowledgeBase("refunds", folder, chunk_size=0)
    assert str(caught.value).splitlines()[:2] == [
        "1 validation error for KnowledgeBase",
        "chunk_size",
    ]
