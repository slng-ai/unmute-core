"""Shared runtime for Unmute voice agents on LiveKit and Pipecat."""

import importlib.metadata
import re

__version__ = importlib.metadata.version("unmute-core")

_EXACT_VERSION = re.compile(r"\d+\.\d+\.\d+")
_LEADING_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class IncompatibleVersionError(ImportError):
    """The installed unmute-core cannot run code generated for another version."""


def _parse(version: str) -> tuple[int, int, int]:
    """Read the leading X.Y.Z, so a local build like "0.1.0.dev3" still parses."""
    match = _LEADING_VERSION.match(version)
    if match is None:
        raise IncompatibleVersionError(f"unmute-core has an unreadable version: {version!r}")
    return int(match[1]), int(match[2]), int(match[3])


def _breaking_part(version: tuple[int, int, int]) -> tuple[int, ...]:
    """Return the part that must match: the major, or 0.MINOR before 1.0."""
    major, minor, _ = version
    return (major,) if major > 0 else (0, minor)


def _next_breaking(version: tuple[int, int, int]) -> str:
    """Return the first version that may break code generated for `version`."""
    major, minor, _ = version
    return f"{major + 1}.0.0" if major > 0 else f"0.{minor + 1}.0"


def require(version: str) -> None:
    """Check that the installed library can run code generated for `version`.

    Generated code calls this at import, so a mismatch fails the deploy, not a call.
    The rule follows protobuf: the installed library is the same major and not older.
    While the version is 0.x, 0.MINOR counts as the major.

    Args:
        version: The exact library version the code was generated for, like "0.2.1".

    Raises:
        ValueError: If `version` is not three numbers like "0.2.1".
        IncompatibleVersionError: If the installed library is older or another major.
    """
    if not _EXACT_VERSION.fullmatch(version):  # own text: it names the form to use, at deploy
        raise ValueError(f"unmute_core.require() needs a version like '0.2.1', got {version!r}")
    installed = _parse(__version__)
    required = _parse(version)
    if _breaking_part(installed) == _breaking_part(required) and installed >= required:
        return
    allowed = f">={version},<{_next_breaking(required)}"
    raise IncompatibleVersionError(
        f"unmute-core {__version__} is installed, but this code needs unmute-core {allowed}. "
        f'Fix: require "unmute-core{allowed}" in pyproject.toml, '
        f"then run: uv lock --upgrade-package unmute-core && uv sync"
    )
