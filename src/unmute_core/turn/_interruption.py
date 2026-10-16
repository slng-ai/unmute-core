"""The checks both frameworks make on an interruption rule, and how words are counted."""

import re
from typing import Annotated, Self

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from unmute_core._pydantic import VALUE, original_errors
from unmute_core.turn._backchannels import Backchannels


@original_errors
@dataclass(frozen=True, slots=True, config=VALUE)
class InterruptionRule:
    """One backchannel rule with the numbers that gate it.

    Attributes:
        backchannels: The rule.
        min_words: Words the caller must say over the agent before it counts.
        min_speech_s: Seconds of speech over the agent before it stops, for ``stop``.
            None leaves the framework's own value.
        min_words_floor: The lowest ``min_words`` the framework takes: 0 on LiveKit, 1 on
            Pipecat.
    """

    backchannels: Backchannels
    min_words: int = 0
    min_speech_s: Annotated[float, Field(ge=0)] | None = None
    min_words_floor: int = 0

    @model_validator(mode="after")
    def _checked(self) -> Self:
        """Refuse too few words for the framework, or a speech time off ``stop``.

        Raises:
            ValueError: The message names the argument and the fix.
        """
        if self.min_words < self.min_words_floor:
            raise ValueError(
                f"min_words must be {self.min_words_floor} or more, got {self.min_words}"
            )
        if self.min_speech_s is not None and self.backchannels != "stop":
            raise ValueError(
                "words already gate the turn, so min_speech_s applies to stop only: "
                f"drop it, or use backchannels='stop' (got {self.backchannels!r})"
            )
        return self


_UNSPACED = (
    (re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"), 2),
    (re.compile(r"[\u0e00-\u0e7f]+"), 4),
)
"""Scripts written without spaces between words, Chinese and Japanese, then Thai, with the
characters one word takes on average."""


def word_count(text: str) -> int:
    """Count the words a caller said, also in a language written without spaces.

    A run of Chinese or Japanese counts one word per two characters, a run of Thai one per
    four. The rest counts what spaces separate.

    Args:
        text: A transcript.
    """
    # ponytail: an average word length, not a segmenter; a dictionary segmenter if it misfires
    count = 0
    for script, size in _UNSPACED:
        count += sum(-(-len(run) // size) for run in script.findall(text))
        text = script.sub(" ", text)
    return count + len(text.split())
