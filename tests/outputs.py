"""The models test doubles share: a result of any shape, and an empty input."""

from pydantic import BaseModel, ConfigDict


class AnyResult(BaseModel):
    """A result of any shape, kept as it is."""

    model_config = ConfigDict(extra="allow")


class NoArguments(BaseModel):
    """A tool input with no arguments."""
