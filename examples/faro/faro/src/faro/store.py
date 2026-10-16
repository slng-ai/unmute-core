"""Faro's learners, kept in one SQLite file so progress survives a call and a restart.

Each learner is one row: their phone number, and their profile and progress as one JSON
document, validated as ``Learner`` on every read. Every write is one short transaction.
The file is ``FARO_DB``, or ``faro.sqlite3`` in the working folder. Three learners are
seeded the first time the file is made:

| Phone | Name | Explains in | Learns | Where they stopped |
|---|---|---|---|---|
| +34600111222 | Lucía | Spanish | English, travel | the second exercise of "ordering at a café" |
| +34600333444 | Diego | Spanish | English, work | the start of "describing a project" |
| +34600555666 | Emma | English | Spanish, daily life | the start of "shopping" |
"""

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, Field

from faro import curriculum
from faro.curriculum import Goal, Lang, Level

Outcome = Literal["independent", "assisted", "not_yet", "insufficient_evidence"]


class Attempt(BaseModel):
    """One answer to one exercise, and what it showed."""

    item: str
    answer: str
    result: Outcome


class Fact(BaseModel):
    """Something the learner told Faro about themselves, kept across calls."""

    key: str = Field(max_length=40)
    text: str = Field(max_length=200)


class Learner(BaseModel):
    """One learner: who they are, their plan, and how far they got.

    Attributes:
        lesson: The index of the lesson under way in the plan's lessons.
        item: The index of the exercise waiting in that lesson.
        completed: The ids of the lessons finished.
        review: The ids of the exercises practised with help or not yet right.
    """

    id: int
    phone: str
    name: str = Field(max_length=40)
    explanation: Lang
    target: Lang
    goal: Goal
    level: Level
    minutes: int = Field(default=10, ge=5, le=60)
    lesson: int = 0
    item: int = 0
    completed: list[str] = Field(default_factory=list)
    review: list[str] = Field(default_factory=list)
    attempts: list[Attempt] = Field(default_factory=list)
    memory: list[Fact] = Field(default_factory=list)

    @property
    def lessons(self) -> tuple[curriculum.Lesson, ...]:
        """The plan's lessons, in order."""
        return curriculum.lessons(self.target, self.goal)

    @property
    def finished(self) -> bool:
        """Whether every lesson of the plan is done."""
        return self.lesson >= len(self.lessons)

    @property
    def waiting(self) -> str:
        """The id of the exercise waiting, or empty once the plan is finished."""
        return "" if self.finished else self.lessons[self.lesson].items[self.item].id


def _seed(
    phone: str, name: str, explanation: Lang, target: Lang, goal: Goal, **rest: object
) -> dict:
    """Return one seeded learner's document."""
    return {
        "phone": phone,
        "name": name,
        "explanation": explanation,
        "target": target,
        "goal": goal,
        **rest,
    }


SEEDS = (
    _seed(
        "+34600111222",
        "Lucía",
        "ES",
        "EN",
        "travel",
        level="A2",
        lesson=1,
        item=1,
        completed=["en-travel-1"],
        memory=[{"key": "hobby", "text": "Enjoys hiking."}],
    ),
    _seed(
        "+34600333444",
        "Diego",
        "ES",
        "EN",
        "work",
        level="B1",
        lesson=2,
        completed=["en-work-1", "en-work-2"],
        review=["en-work-2-1"],
        memory=[{"key": "hobby", "text": "Enjoys cooking."}],
    ),
    _seed(
        "+34600555666",
        "Emma",
        "EN",
        "ES",
        "daily",
        level="A1",
        lesson=1,
        completed=["es-daily-1"],
        memory=[{"key": "prompts", "text": "Prefers short prompts."}],
    ),
)
"""Invented people. Any resemblance to a real one is a coincidence."""


class Store:
    """The learners, in one SQLite file. Thread-safe: tools may run in threads."""

    def __init__(self, path: str | Path = "faro.sqlite3") -> None:
        """Open the file, made with the seeded learners the first time.

        Args:
            path: The SQLite file, or ``:memory:`` for a store that lives as long as this object.
        """
        self._lock = Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS learners ("
                "id INTEGER PRIMARY KEY, phone TEXT UNIQUE NOT NULL, body TEXT NOT NULL)"
            )
            for seed in SEEDS:
                self._db.execute(
                    "INSERT OR IGNORE INTO learners (phone, body) VALUES (?, ?)",
                    (seed["phone"], json.dumps(seed)),
                )

    def _learner(self, row: tuple[int, str] | None) -> Learner | None:
        """Return the learner a row holds, checked."""
        if row is None:
            return None
        return Learner.model_validate({**json.loads(row[1]), "id": row[0]})

    def find(self, phone: str) -> Learner | None:
        """Return the learner with this phone number, or None."""
        with self._lock:
            row = self._db.execute(
                "SELECT id, body FROM learners WHERE phone = ?", (phone,)
            ).fetchone()
        return self._learner(row)

    def get(self, learner_id: int) -> Learner:
        """Return one learner by id.

        Raises:
            KeyError: When no learner has that id.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT id, body FROM learners WHERE id = ?", (learner_id,)
            ).fetchone()
        if (found := self._learner(row)) is None:
            raise KeyError(learner_id)
        return found

    def create(
        self, phone: str, name: str, explanation: Lang, target: Lang, goal: Goal, level: Level
    ) -> Learner:
        """Create a learner with a fresh plan.

        Raises:
            ValueError: When a learner already has that phone number.
        """
        body = _seed(phone, name, explanation, target, goal, level=level)
        with self._lock:
            try:
                cursor = self._db.execute(
                    "INSERT INTO learners (phone, body) VALUES (?, ?)", (phone, json.dumps(body))
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("a profile with that phone number already exists") from error
        return Learner.model_validate({**body, "id": cursor.lastrowid})

    def save(self, learner: Learner) -> None:
        """Write a learner back, whole."""
        body = learner.model_dump(mode="json", exclude={"id"})
        with self._lock:
            self._db.execute(
                "UPDATE learners SET body = ? WHERE id = ?", (json.dumps(body), learner.id)
            )

    def count(self) -> int:
        """Return how many learners there are."""
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM learners").fetchone()[0]
