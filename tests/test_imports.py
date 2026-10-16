"""Every subpackage imports with an empty environment, and only loads what it needs.

Each check runs in a fresh interpreter, because an import done once stays done.
"""

import subprocess
import sys
import textwrap

import pytest


def run_clean(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with no environment variables at all."""
    return subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code)],
        env={},
        capture_output=True,
        text=True,
        check=False,
    )


def test_every_subpackage_imports_with_an_empty_environment() -> None:
    result = run_clean("""
        import importlib, importlib.util, pkgutil, unmute_core
        extras = {"livekit": "livekit", "pipecat": "pipecat", "knowledge": "llama_index"}
        missing = {part for part, package in extras.items() if not importlib.util.find_spec(package)}
        for module in pkgutil.walk_packages(unmute_core.__path__, "unmute_core.", lambda _: None):
            if module.name.split(".")[1] not in missing:
                importlib.import_module(module.name)
    """)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("package", ["unmute_core.slng", "unmute_core.handoff"])
def test_shared_names_do_not_load_the_tasks_package(package: str) -> None:
    result = run_clean(f"""
        import sys, {package}
        assert "unmute_core.tasks" not in sys.modules, "loaded unmute_core.tasks"
    """)
    assert result.returncode == 0, result.stderr


def test_slng_headers_and_settings_work_without_openai() -> None:
    result = run_clean("""
        import sys
        sys.modules["openai"] = None
        from unmute_core.slng import RouterHeaders, SlngSettings
        RouterHeaders(agent_id="a", session_id="s")
        assert "unmute_core.slng._router" not in sys.modules
        try:
            from unmute_core.slng import RouterModelCall
        except ImportError as error:
            print(error)
    """)
    assert result.returncode == 0, result.stderr
    assert 'uv add "unmute-core[slng]"' in result.stdout


@pytest.mark.parametrize(
    ("extra", "package"),
    [("livekit", "livekit"), ("pipecat", "pipecat"), ("knowledge", "llama_index")],
)
def test_a_missing_extra_names_itself(extra: str, package: str) -> None:
    result = run_clean(f"""
        import sys
        sys.modules["{package}"] = None
        import unmute_core.{extra}
    """)
    assert result.returncode != 0
    assert f'uv add "unmute-core[{extra}]"' in result.stderr
