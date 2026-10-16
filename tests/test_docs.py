"""The docs pages keep the rules the Unmute site checks, so they copy into it as they are."""

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[1] / "docs"
PAGES = sorted((DOCS / "core").rglob("*.mdx"))
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
VERSION = re.compile(r"\bv\d+\.\d+\.\d+\b")


def _nav_pages(node: object) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _nav_pages(item)
    elif isinstance(node, dict):
        if "root" in node:
            yield node["root"]
        for key in ("tabs", "groups", "pages"):
            yield from _nav_pages(node.get(key, []))


def _prose(page: Path) -> str:
    return FENCE.sub("", page.read_text())


def test_every_page_is_in_the_navigation_once() -> None:
    listed = list(_nav_pages(json.loads((DOCS / "docs.json").read_text())["navigation"]))
    on_disk = [str(page.relative_to(DOCS).with_suffix("")) for page in PAGES]
    assert sorted(listed) == sorted(on_disk)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(DOCS)))
def test_a_page_has_a_title_and_a_description(page: Path) -> None:
    head = FRONTMATTER.match(page.read_text())
    assert head is not None, f"{page.name} has no frontmatter"
    assert re.search(r"^title: \S", head[1], re.MULTILINE)
    assert re.search(r"^description: \S", head[1], re.MULTILINE)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(DOCS)))
def test_a_page_has_no_dash_punctuation_and_no_h1(page: Path) -> None:
    prose = _prose(page)
    assert "\u2014" not in prose and "\u2013" not in prose, "use a full stop, not a dash"
    assert not re.search(r"^# ", prose, re.MULTILINE), "the title is the H1"


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(DOCS)))
def test_a_page_states_no_release_version(page: Path) -> None:
    assert not VERSION.search(page.read_text()), "point at the changelog, not a version"
