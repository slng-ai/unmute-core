import importlib.metadata

import pytest

import unmute_core
from unmute_core import IncompatibleVersionError, require


def test_version_comes_from_package_metadata():
    """__version__ is read from the installed package, not hard-coded."""
    assert unmute_core.__version__ == importlib.metadata.version("unmute-core")


@pytest.mark.parametrize(
    ("installed", "required", "ok"),
    [
        ("1.2.3", "1.2.3", True),  # same version
        ("1.2.4", "1.2.3", True),  # newer patch
        ("1.3.0", "1.2.3", True),  # newer minor, same major
        ("1.2.3.dev4", "1.2.3", True),  # local build suffix
        ("0.2.5", "0.2.1", True),  # newer patch at 0.x
        ("0.3.0", "0.2.1", False),  # newer minor at 0.x is another major
        ("1.2.2", "1.2.3", False),  # older patch
        ("1.1.9", "1.2.0", False),  # older minor
        ("2.0.0", "1.2.3", False),  # newer major
        ("1.9.9", "2.0.0", False),  # older major
    ],
)
def test_require(monkeypatch, installed, required, ok):
    """Same major and not older passes. Anything else is refused."""
    monkeypatch.setattr(unmute_core, "__version__", installed)
    if ok:
        require(required)
    else:
        with pytest.raises(IncompatibleVersionError):
            require(required)


def test_require_message_names_versions_and_fix(monkeypatch):
    """The error names both versions, the allowed range and the fix command."""
    monkeypatch.setattr(unmute_core, "__version__", "0.1.3")
    with pytest.raises(IncompatibleVersionError) as excinfo:
        require("0.2.0")
    message = str(excinfo.value)
    assert "unmute-core 0.1.3 is installed" in message
    assert ">=0.2.0,<0.3.0" in message
    assert '"unmute-core>=0.2.0,<0.3.0" in pyproject.toml' in message
    assert "uv lock --upgrade-package unmute-core" in message


def test_require_upper_bound_after_one_dot_zero(monkeypatch):
    """From 1.0 on, the next major is the upper bound."""
    monkeypatch.setattr(unmute_core, "__version__", "1.0.0")
    with pytest.raises(IncompatibleVersionError, match=r">=1\.4\.0,<2\.0\.0"):
        require("1.4.0")


def test_require_is_an_import_error(monkeypatch):
    """A mismatch fails the import of the generated module."""
    monkeypatch.setattr(unmute_core, "__version__", "0.0.1")
    with pytest.raises(ImportError):
        require("0.0.2")


@pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "1.2.3rc1", "", "1.2.3 "])
def test_require_rejects_bad_text(bad):
    """Generated code must pass an exact X.Y.Z version."""
    with pytest.raises(ValueError, match="needs a version like"):
        require(bad)
