"""Search over a folder of documents. Needs the ``[knowledge]`` extra.

``KnowledgeBase(name, folder, embed=...)`` holds one base. ``bake()`` writes its index at
image build time, ``start()`` loads it at process start, ``look_up(query)`` runs in a call.
"""

try:
    from unmute_core.knowledge._base import KnowledgeBase, KnowledgeError, Mode
except ModuleNotFoundError as error:
    if not (error.name or "").startswith("llama_index"):
        raise
    raise ImportError(
        'unmute_core.knowledge needs the knowledge extra. Install it: uv add "unmute-core[knowledge]"'
    ) from error

__all__ = ["KnowledgeBase", "KnowledgeError", "Mode"]
