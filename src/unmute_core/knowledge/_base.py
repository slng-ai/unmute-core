"""Search over one folder of documents. Bake the index at build time, look up during a call.

Ported from unmute@d0c67f1 internal/generate/templates/knowledge/knowledge.py.tmpl, which
emits one module per agent. Here each base is one ``KnowledgeBase`` with its settings as
constructor arguments.

The content is fixed at start: a lookup embeds the question and searches. It never reads a
file and never re-indexes. Only ``bake`` writes, and only at image build time.
"""

import json
import logging
import re
import time
from collections.abc import Callable
from itertools import islice, zip_longest
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from pydantic import BaseModel, ConfigDict, Field, model_validator

from unmute_core.settings import SettingsError, env_value
from unmute_core.state import CallState
from unmute_core.tools import ToolSpec

logger = logging.getLogger("unmute_core.knowledge")

Mode = Literal["meaning", "keyword", "hybrid"]
MODES: tuple[str, ...] = get_args(Mode)
MAX_CHUNK_SIZE = 2048
MAX_TOP_K = 20


class _Hit(BaseModel):
    """One search hit: node id, passage text, source file name, and a score (None for keyword)."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    source: str
    score: float | None = None


class _Passage(BaseModel):
    """One passage a lookup returns. A keyword hit has no score."""

    model_config = ConfigDict(frozen=True)

    text: str
    source: str
    score: float | None = None


class _Results(BaseModel):
    """What a lookup that searched returns."""

    model_config = ConfigDict(frozen=True)

    results: list[_Passage]


class _Settings(BaseModel):
    """The settings of one base, checked once when the base is built."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, title="KnowledgeBase")

    name: str
    mode: Mode
    chunk_size: Annotated[int, Field(ge=1, le=MAX_CHUNK_SIZE)]
    chunk_overlap: Annotated[int, Field(ge=0)]
    top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)]
    min_score: Annotated[int | float, Field(ge=0, le=1)]
    embed: Callable[[], BaseEmbedding] | None
    credential_env: str | None

    @model_validator(mode="after")
    def _fit_together(self) -> Self:
        """Check the settings that depend on each other.

        Raises:
            ValueError: Naming the base and the setting to fix.
        """
        where = f"knowledge {self.name!r}"
        if self.chunk_overlap > self.chunk_size:
            raise ValueError(
                f"{where}: chunk_overlap {self.chunk_overlap} must be 0 to chunk_size "
                f"({self.chunk_size})"
            )
        if self.mode == "keyword" and self.min_score:
            raise ValueError(f"{where}: keyword hits have no score, so min_score must be 0")
        if (self.mode != "keyword") != (self.embed is not None):
            raise ValueError(
                f"{where}: mode {self.mode!r} needs an embedder"
                if self.embed is None
                else f"{where}: keyword mode embeds nothing, so leave out embed"
            )
        return self


_PDF_MAGIC = "%PDF-"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


RELEVANCE = (
    "The results may not answer the question. They are ordered best first, and each carries a "
    "relevance score where a higher number means a closer match. If nothing returned actually "
    "answers what was asked, say you do not have that information rather than offering the "
    "closest result."
)
"""What the lookup tool's description ends with, so the model can decline a poor match."""


class _LookUp:
    """Runs a base's ``look_up`` as a tool."""

    def __init__(self, base: "KnowledgeBase") -> None:
        """Keep the base to search.

        Args:
            base: The base a call searches.
        """
        self._base = base

    async def run(self, arguments: dict[str, Any], state: CallState | None) -> object:
        """Search the base for the model's ``query``.

        Args:
            arguments: The model's arguments, holding ``query``.
            state: The call state, which a lookup does not read.

        Returns:
            What ``KnowledgeBase.look_up`` returns.
        """
        return await self._base.look_up(arguments["query"])


class KnowledgeError(RuntimeError):
    """A startup failure. The process should stop rather than serve calls knowing nothing."""


def _unavailable() -> dict[str, Any]:
    """The lookup result that tells the model it cannot check right now."""
    return {"error": "lookup unavailable"}


def _clean(text: str) -> str:
    """Remove the control characters a PDF reader leaves for bullets, and collapse spacing.

    Newlines and tabs stay, because the splitter finds sentence boundaries with them.
    """
    return re.sub(r"[ \t]{2,}", " ", _CONTROL.sub("", text))


def _source_name(metadata: dict[str, Any]) -> str:
    """Return a document's file name, never its path. The agent may say it out loud."""
    return Path(metadata.get("file_name") or metadata.get("file_path") or "").name


def _keyword_retriever(nodes: list[BaseNode], top_k: int) -> BM25Retriever:
    """Return the keyword retriever, asking for no more passages than the base holds.

    Older BM25 releases raise when ``top_k`` is larger than the number of passages.
    """
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(top_k, len(nodes)))


def _merge(meaning: list[_Hit], keyword: list[_Hit], top_k: int) -> list[_Hit]:
    """Interleave the two halves, meaning first at each step, drop repeats, then truncate.

    Interleaved, not one after the other: each half returns ``top_k`` hits, so the first half
    would fill every slot. Measured on 15 questions built to defeat meaning search:
    concatenating scored 14/15 and interleaving 15/15.
    """
    unique: dict[str, _Hit] = {}
    for pair in zip_longest(meaning, keyword):
        for hit in pair:
            if hit is not None:
                unique.setdefault(hit.id, hit)
    return list(islice(unique.values(), top_k))


def _hits(found: list[NodeWithScore], *, scored: bool) -> list[_Hit]:
    """Turn retrieved nodes into hits. BM25 scores are on another scale, so they are dropped."""
    return [
        _Hit(
            id=hit.node.node_id,
            text=hit.node.get_content(),
            source=_source_name(hit.node.metadata or {}),
            score=hit.score if scored else None,
        )
        for hit in found
    ]


class KnowledgeBase:
    """One folder of documents, searchable by meaning, by keyword, or both.

    Example:
        ``KnowledgeBase("refunds", Path("knowledge/refunds"), embed=lambda: OpenAIEmbedding())``

    Attributes:
        name: The base name, used in log lines and errors.
        documents: The folder holding the documents (``.txt``, ``.md``, ``.pdf``).
        index_dir: Where ``bake`` writes the index and ``start`` looks for it. None never bakes.
        mode: ``meaning`` (embeddings), ``keyword`` (BM25), or ``hybrid`` (both, interleaved).
        chunk_size: Passage size in tokens. 90 tokens is about 65 words.
        chunk_overlap: Tokens shared by neighbouring passages.
        top_k: How many passages a lookup returns.
        min_score: Drop meaning hits below this similarity. 0 keeps all. Keyword hits have no
            score and are always kept.
        embed: Builds the llama-index embedder. Needed exactly when the mode embeds.
        credential_env: The environment variable the embedder needs. Checked before the
            first embedding call, so a missing key fails at start and names the variable.
    """

    def __init__(
        self,
        name: str,
        documents: Path,
        *,
        index_dir: Path | None = None,
        mode: Mode = "hybrid",
        chunk_size: int = 90,
        chunk_overlap: int = 20,
        top_k: int = 3,
        min_score: float = 0.0,
        embed: Callable[[], BaseEmbedding] | None = None,
        credential_env: str | None = None,
    ) -> None:
        """Check the settings. Nothing is read or embedded until ``start`` or ``bake``.

        Args:
            name: The base name, used in log lines and errors.
            documents: The folder holding the documents.
            index_dir: Where ``bake`` writes the index and ``start`` looks for it.
            mode: ``meaning``, ``keyword`` or ``hybrid``.
            chunk_size: Passage size in tokens, 1 to 2048.
            chunk_overlap: Tokens shared by neighbouring passages, up to ``chunk_size``.
            top_k: How many passages a lookup returns, 1 to 20.
            min_score: Drop meaning hits below this similarity, 0 to 1.
            embed: Builds the llama-index embedder. Needed exactly when the mode embeds.
            credential_env: The environment variable the embedder needs.

        Raises:
            ValueError: If a setting is out of range, or ``embed`` does not match the mode.
        """
        checked = _Settings(
            name=name,
            mode=mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            min_score=min_score,
            embed=embed,
            credential_env=credential_env,
        )
        self.name = checked.name
        self.documents = documents
        self.index_dir = index_dir
        self.mode = checked.mode
        self.chunk_size = checked.chunk_size
        self.chunk_overlap = checked.chunk_overlap
        self.top_k = checked.top_k
        self.min_score = checked.min_score
        self.embed = embed
        self.credential_env = credential_env
        self._vector: BaseRetriever | None = None
        self._keyword: BM25Retriever | None = None
        self._started = False

    def bake(self) -> None:
        """Build the index and write it to ``index_dir``. Run at image build time only.

        ``baked.json`` is written last, so a half-written index never reads as a whole one.

        Raises:
            KnowledgeError: If there is no ``index_dir``, or the documents cannot be indexed.
        """
        if self.index_dir is None:
            raise KnowledgeError(f"knowledge {self.name!r}: bake needs an index_dir to write to")
        embed_model = self._embedder()
        nodes = self._nodes()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.mode != "meaning":
            _keyword_retriever(nodes, self.top_k).persist(str(self.index_dir / "keyword"))
        if self.mode != "keyword":
            VectorStoreIndex(
                nodes=nodes, embed_model=embed_model, show_progress=False
            ).storage_context.persist(persist_dir=str(self.index_dir / "vector"))
        (self.index_dir / "baked.json").write_text(
            json.dumps({"settings": self._fingerprint(), "passages": len(nodes)}, sort_keys=True)
        )
        logger.info("knowledge %r: baked %d passages (mode %s)", self.name, len(nodes), self.mode)

    def start(self) -> None:
        """Make the base searchable. Call once, at process start, off the call path.

        Loads the baked index when ``index_dir`` holds one. Otherwise it reads and embeds the
        documents in memory, which is correct but repeats the cost in every worker process.

        Raises:
            KnowledgeError: If the credential is missing, the baked index was built with other
                settings, or the documents yield no text.
        """
        embed_model = self._embedder()
        baked = self.index_dir / "baked.json" if self.index_dir else None
        if baked is not None and baked.is_file():
            self._load(baked, embed_model)
        else:
            logger.warning(
                "knowledge %r: no baked index, embedding the corpus at startup instead. This "
                "is correct but costs every worker process the full build; call bake() at "
                "image build time to avoid it.",
                self.name,
            )
            self._build(embed_model)
        self._started = True

    def tool(self, name: str = "look_up", description: str = "") -> ToolSpec:
        """Return the tool a model calls to search this base.

        Ported from unmute@b9b35b5 internal/generate/knowledge.go.

        Args:
            name: The tool's name.
            description: When to look up, for the model. A note on how to read the results
                follows it.

        Returns:
            A tool that runs ``look_up`` with the model's ``query``.
        """
        return ToolSpec(
            name=name,
            description=f"{description}\n\n{RELEVANCE}" if description else RELEVANCE,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up. Use the caller's own words.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            output=_Results,
            executor=_LookUp(self),
        )

    async def look_up(self, query: str) -> dict[str, Any]:
        """Search the base and return at most ``top_k`` passages. Never raises.

        Three shapes, so the model can tell them apart:
        ``{"results": [{"text", "source", "score"?}, ...]}`` found something,
        ``{"results": []}`` found nothing, ``{"error": "lookup unavailable"}`` could not search.
        No query and no passage ever goes to the log.

        Args:
            query: The caller's question, in their own words.

        Returns:
            One of the three shapes above.
        """
        started = time.monotonic()
        if not self._started:
            logger.error("knowledge %r: looked up before start()", self.name)
            return _unavailable()
        try:
            meaning = (
                _hits(await self._vector.aretrieve(query), scored=True) if self._vector else []
            )
            keyword = (
                _hits(await self._keyword.aretrieve(query), scored=False) if self._keyword else []
            )
        except Exception:
            logger.exception("knowledge %r: lookup failed", self.name)
            return _unavailable()
        merged = _merge(meaning, keyword, self.top_k)
        kept = [
            hit
            for hit in merged
            if not (self.min_score and hit.score is not None and hit.score < self.min_score)
        ]
        if dropped := len(merged) - len(kept):
            logger.info(
                "knowledge %r: dropped %d result(s) below min_score %s",
                self.name,
                dropped,
                self.min_score,
            )
        results = [
            _Passage(
                text=hit.text,
                source=hit.source,
                score=None if hit.score is None else round(hit.score, 3),
            )
            for hit in kept
        ]
        logger.info(
            "knowledge %r: returned %d result(s) in %d ms",
            self.name,
            len(results),
            int((time.monotonic() - started) * 1000),
        )
        return _Results(results=results).model_dump(exclude_none=True)

    def _fingerprint(self) -> str:
        """Return the settings that decide what the passages are, as a stable string."""
        return json.dumps(
            {
                "mode": self.mode,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "min_score": self.min_score,
            },
            sort_keys=True,
        )

    def _embedder(self) -> BaseEmbedding | None:
        """Build the embedder, after checking its credential. None when the mode embeds nothing.

        Raises:
            KnowledgeError: If the credential is not set.
        """
        if self.embed is None:
            return None
        if self.credential_env and not self._credential_set():
            raise KnowledgeError(
                f"knowledge {self.name!r}: {self.credential_env} is not set, so documents "
                "cannot be embedded"
            )
        return self.embed()

    def _credential_set(self) -> bool:
        """Return True when the credential variable holds a value.

        Raises:
            KnowledgeError: If the variable holds something that cannot be read.
        """
        try:
            return env_value(str(self.credential_env), str) is not None
        except SettingsError as error:
            raise KnowledgeError(f"knowledge {self.name!r}: {error}") from error

    def _read(self) -> list[Document]:
        """Read the documents, failing loudly on a missing folder or an empty read.

        ``exclude_hidden=False`` matters: by default the reader silently returns nothing for
        a path with a dot-component in it, which a checkout path often has.

        Raises:
            KnowledgeError: If the folder is missing or holds no documents.
        """
        if not self.documents.is_dir():
            raise KnowledgeError(f"knowledge {self.name!r}: {self.documents} is not a folder")
        try:
            documents = SimpleDirectoryReader(str(self.documents), exclude_hidden=False).load_data()
        except ValueError as error:
            logger.warning(
                "knowledge %r: the reader refused %s: %s", self.name, self.documents, error
            )
            documents = []
        if not documents:
            raise KnowledgeError(
                f"knowledge {self.name!r}: read no documents from {self.documents}"
            )
        return documents

    def _with_text(self, documents: list[Document]) -> list[Document]:
        """Keep the documents that have text, cleaned, with only their file name as metadata.

        The file name is hidden from the embedding and the model. The splitter subtracts the
        metadata's length from ``chunk_size``, so a long path would make small chunks fail.

        Raises:
            KnowledgeError: If a PDF was read as raw bytes, or no document has any text.
        """
        kept: list[Document] = []
        empty: list[str] = []
        for document in documents:
            text = document.text or ""
            source = _source_name(document.metadata or {})
            if text.lstrip().startswith(_PDF_MAGIC):
                raise KnowledgeError(
                    f"knowledge {self.name!r}: {source} was read as raw PDF bytes, not as text. "
                    "llama-index-readers-file is missing, so the reader fell back to plain text. "
                    "Install unmute-core[knowledge]."
                )
            if not text.strip():
                empty.append(source)
                continue
            document.set_content(_clean(text))
            document.metadata = {"file_name": source}
            document.excluded_embed_metadata_keys = ["file_name"]
            document.excluded_llm_metadata_keys = ["file_name"]
            kept.append(document)
        for source in empty:
            logger.warning("knowledge %r: skipping %s, no text layer", self.name, source)
        if not kept:
            raise KnowledgeError(
                f"knowledge {self.name!r}: no document yielded any text ({', '.join(empty)}). "
                "A PDF with no text layer needs OCR before it can be searched."
            )
        logger.info(
            "knowledge %r: %d of %d documents indexed", self.name, len(kept), len(documents)
        )
        return kept

    def _nodes(self) -> list[BaseNode]:
        """Read the documents and split them into passages."""
        return SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        ).get_nodes_from_documents(self._with_text(self._read()))

    def _build(self, embed_model: BaseEmbedding | None) -> None:
        """Build the retrievers in memory from the documents."""
        nodes = self._nodes()
        if self.mode != "meaning":
            self._keyword = _keyword_retriever(nodes, self.top_k)
        if self.mode != "keyword":
            self._vector = VectorStoreIndex(
                nodes=nodes, embed_model=embed_model, show_progress=False
            ).as_retriever(similarity_top_k=self.top_k)
        logger.info(
            "knowledge %r: %d passages indexed at startup (mode %s, chunk_size %d, overlap %d, "
            "top_k %d)",
            self.name,
            len(nodes),
            self.mode,
            self.chunk_size,
            self.chunk_overlap,
            self.top_k,
        )

    def _load(self, baked: Path, embed_model: BaseEmbedding | None) -> None:
        """Load the baked retrievers. Reads only, and embeds nothing of the corpus.

        Raises:
            KnowledgeError: If the index was baked with other settings.
        """
        recorded = json.loads(baked.read_text())
        if recorded.get("settings") != self._fingerprint():
            raise KnowledgeError(
                f"knowledge {self.name!r}: the baked index was built with other settings. "
                f"Baked: {recorded.get('settings')}. Now: {self._fingerprint()}. "
                "Bake it again rather than answer from passages that do not match."
            )
        folder = baked.parent
        if self.mode != "meaning":
            self._keyword = BM25Retriever.from_persist_dir(str(folder / "keyword"))
            self._keyword.similarity_top_k = min(self.top_k, recorded["passages"])
        if self.mode != "keyword":
            self._vector = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(folder / "vector")),
                embed_model=embed_model,
            ).as_retriever(similarity_top_k=self.top_k)
        logger.info(
            "knowledge %r: %s passages loaded from the baked index, nothing embedded "
            "(mode %s, top_k %d)",
            self.name,
            recorded.get("passages"),
            self.mode,
            self.top_k,
        )
