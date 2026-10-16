"""``PromptSuffix``: literal text added after every system prompt a model reads.

Some models take an instruction no request parameter carries, like Qwen3's ``/no_think``.
The rules and wording follow Unmute's ``prompt_suffix`` (unmute@2c78a5a
internal/ir/build.go:813-818, internal/ir/validate.go:3117-3160).
"""

import re
from dataclasses import field
from typing import Annotated

from pydantic import AfterValidator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.hooks._hooks import PromptInfo

PROMPT_SUFFIX_MAX = 512
"""The longest suffix. It is one directive, not a second prompt."""
_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]*?)\s*\}\}")


def check_prompt_suffix(text: str) -> str:
    """Return the suffix stripped, after Unmute's checks.

    Raises:
        ValueError: If it is blank, longer than 512 characters, or holds a ``{{...}}``.
    """
    if not text.strip():
        raise ValueError(
            "prompt_suffix is empty or whitespace: leave it out to mean off, so a typo cannot "
            "read as a deliberate blank"
        )
    if len(text) > PROMPT_SUFFIX_MAX:
        raise ValueError(
            f"prompt_suffix is {len(text)} characters and the bound is {PROMPT_SUFFIX_MAX}: it "
            "is a directive added to every prompt this model reads, not a second prompt. Put "
            "the wording in the prompt"
        )
    if _PLACEHOLDER.search(text):
        raise ValueError(
            "prompt_suffix contains a {{...}} placeholder: the router is given the values the "
            "prompts name, this name would not be among them, and the request fails mid-call. "
            "Use literal text"
        )
    return text.strip()


def add_prompt_suffix(prompt: str, suffix: str) -> str:
    """Return the prompt with the suffix after it, on its own after a blank line."""
    return prompt.rstrip("\n") + "\n\n" + suffix


@original_errors
@dataclass(frozen=True, config=VALUE)
class PromptSuffix:
    """A ``prompt`` hook that adds the same text after every system prompt.

    Add it with ``hooks.add(PromptSuffix("/no_think"))``. It reaches the agent's prompt,
    each task step's, the finish checks' and the summaries'.

    Attributes:
        text: The suffix, like ``/no_think``. Stripped. It must follow ``check_prompt_suffix``.
        model: Only prompts this model reads get it, like a summarizer's. None means all.
    """

    text: Annotated[str, AfterValidator(check_prompt_suffix)]
    model: str | None = field(default=None, kw_only=True)

    def prompt(self, text: str, info: PromptInfo) -> str:
        """Add the suffix, unless the prompt goes to another model."""
        if self.model is not None and info.model != self.model:
            return text
        return add_prompt_suffix(text, self.text)
