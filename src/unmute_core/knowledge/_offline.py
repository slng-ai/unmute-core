"""A stand-in for OpenAI embeddings, so an offline smoke run builds its bases with no network."""

from typing import Any


def mock_openai_embeddings() -> None:
    """Make llama-index's OpenAI embedder its mock embedder, when both are installed."""
    try:
        import llama_index.embeddings.openai as openai_embeddings  # ty: ignore[unresolved-import]
        from llama_index.core.embeddings import MockEmbedding
    except ImportError:
        return

    def mock(*args: Any, **kwargs: Any) -> MockEmbedding:
        return MockEmbedding(embed_dim=8)

    openai_embeddings.OpenAIEmbedding = mock
