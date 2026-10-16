"""The core imports no framework. Only its own adapter package may import one.

The same holds for llama-index, which only the ``[knowledge]`` package imports.
"""

import ast
from pathlib import Path

import unmute_core

ALLOWED_IN = {"livekit": "livekit", "pipecat": "pipecat", "llama_index": "knowledge"}
ADAPTERS = ("livekit", "pipecat")
ROOT_NAME = "unmute_core"
PACKAGE_ROOT = Path(unmute_core.__file__).parent


def imported_modules(node: ast.Import | ast.ImportFrom, package: tuple[str, ...]) -> list[str]:
    """Return the absolute module names an import statement reaches.

    Args:
        node: An import statement.
        package: The package the importing file sits in, like ``("unmute_core", "livekit")``.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = list(package[: len(package) - node.level + 1]) if node.level else []
    module = ".".join([*base, *(node.module.split(".") if node.module else [])])
    if module == ROOT_NAME:
        return [f"{module}.{alias.name}" for alias in node.names]
    return [module] if module else []


def framework_violations(root: Path) -> list[str]:
    """List imports that cross a boundary.

    A framework, or llama-index, may only be imported by the one package allowed to. The core
    may not import an adapter, and one adapter may not import the other.

    ponytail: only static imports are seen. importlib.import_module("livekit") is not.
    """
    violations = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        adapter = rel.parts[0] if len(rel.parts) > 1 else None
        package = (ROOT_NAME, *rel.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import | ast.ImportFrom):
                violations.extend(
                    f"{rel}:{node.lineno} imports {name}"
                    for name in imported_modules(node, package)
                    if _owner(name) not in (None, adapter)
                )
    return violations


def _owner(name: str) -> str | None:
    """Return the package that alone may import ``name``, or None when any module may."""
    parts = name.split(".")
    if parts[0] == ROOT_NAME:
        return parts[1] if len(parts) > 1 and parts[1] in ADAPTERS else None
    return ALLOWED_IN.get(parts[0])


def test_core_imports_no_framework():
    """No module in unmute_core imports a framework outside its adapter."""
    assert framework_violations(PACKAGE_ROOT) == []


def test_guard_catches_violations(tmp_path):
    """The guard flags planted imports, so the clean result above means something."""
    (tmp_path / "livekit").mkdir()
    (tmp_path / "pipecat").mkdir()
    (tmp_path / "state.py").write_text("import json\nimport livekit.agents\n")
    (tmp_path / "livekit" / "ok.py").write_text("from livekit import agents\n")
    (tmp_path / "livekit" / "mixed.py").write_text(
        "def f():\n    from pipecat.frames import Frame\n"
    )
    (tmp_path / "pipecat" / "ok.py").write_text("import pipecat.flows as flows\n")
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "ok.py").write_text("from llama_index.core import Document\n")
    (tmp_path / "prefetch.py").write_text("import llama_index.core\n")

    (tmp_path / "relative.py").write_text("from .livekit import ok\n")

    assert framework_violations(tmp_path) == [
        "livekit/mixed.py:2 imports pipecat.frames",
        "prefetch.py:1 imports llama_index.core",
        "relative.py:1 imports unmute_core.livekit",
        "state.py:2 imports livekit.agents",
    ]


def test_guard_catches_core_importing_an_adapter_and_adapters_importing_each_other(tmp_path):
    """The core may not import an adapter, absolutely or relatively, nor one adapter the other."""
    for package in ("livekit", "pipecat", "state"):
        (tmp_path / package).mkdir()
    (tmp_path / "livekit" / "ok.py").write_text("from unmute_core.livekit import _agent\n")
    (tmp_path / "livekit" / "core.py").write_text("from unmute_core.state import CallState\n")
    (tmp_path / "state" / "absolute.py").write_text("import unmute_core.pipecat\n")
    (tmp_path / "state" / "relative.py").write_text("from ..livekit import agent\n")
    (tmp_path / "state" / "dotted.py").write_text("from .. import pipecat\n")
    (tmp_path / "livekit" / "other.py").write_text("from unmute_core.pipecat import tasks\n")
    (tmp_path / "pipecat" / "other.py").write_text("from ..livekit import tasks\n")
    (tmp_path / "pipecat" / "same.py").write_text("from . import tasks\nfrom .tasks import x\n")

    assert framework_violations(tmp_path) == [
        "livekit/other.py:1 imports unmute_core.pipecat",
        "pipecat/other.py:1 imports unmute_core.livekit",
        "state/absolute.py:1 imports unmute_core.pipecat",
        "state/dotted.py:1 imports unmute_core.pipecat",
        "state/relative.py:1 imports unmute_core.livekit",
    ]
