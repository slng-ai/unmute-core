# Contributing

Thank you for helping. One pull request makes one change.

## Set up

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for you.

```bash
uv sync --all-extras
```

## Check your change

Run these before you open a pull request. CI runs the same.

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uvx ty check
scripts/check-examples.sh
```

## Rules

1. The core never imports LiveKit or Pipecat. Only `unmute_core/livekit/` and `unmute_core/pipecat/` do. A test checks this.
2. A behaviour a caller hears or the model sees gets a JSON scenario in `tests/conformance/scenarios/`. Both adapters run it, so the behaviour stays the same on both frameworks.
3. Core logic gets a row in an existing table test, through the public API.
4. Tests never need real API keys. Tests that do are marked `live`. They run in the nightly workflow.
5. Every class and function in `src/` has a Google-style docstring and type hints.
6. A change users can notice gets a line in `CHANGELOG.md`, under the next version. A change that can break users goes under "Changes that can break you".

[CLAUDE.md](CLAUDE.md) holds the full design rules and the Python style.

## Releases

Maintainers release by pushing a tag `vX.Y.Z` that matches `version` in `pyproject.toml`. The release workflow runs CI and the live tests, publishes to TestPyPI, then waits for a maintainer to approve the upload to PyPI.
