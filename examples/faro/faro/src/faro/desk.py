"""Faro's tools: find the learner, sign them up, and run the lesson.

Every tool returns a typed result with ``said``: the exact line code says next, in the
learner's explanation language. The registry in ``faro.agent`` saves the result's values
and speaks ``said``, so the model words only its own feedback and hints.

The model's part in a lesson is one typed call, ``grade_answer``: the learner's answer,
a grade from four, and one line of feedback. Code turns the grade into a result, records
the attempt, moves the learner on or not, and says the next exercise. A model cannot
complete a lesson by saying so: only graded answers move progress.
"""

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from unmute_core.state import Phone
from unmute_core.tools import ToolError

from faro import curriculum
from faro.curriculum import Goal, Lang, Level
from faro.store import Attempt, Fact, Learner, Outcome, Store

Grade = Literal["correct", "partly", "wrong", "unclear"]
RESULTS: dict[str, Outcome] = {
    "correct": "independent",
    "partly": "not_yet",
    "wrong": "not_yet",
    "unclear": "insufficient_evidence",
}
"""What each grade records. A correct answer after a hint records ``assisted`` instead."""


NumberSaid = Annotated[
    Phone,
    Field(
        description="The whole number the caller said, with its country code, like +34 600 111 222."
    ),
]


class Identified(BaseModel):
    """What finding or creating a learner returns: who they are, and code's next line."""

    status: Literal["known", "new", "created", "foreign"]
    said: str
    phone: str
    explanation: Lang
    learner_id: int | None = None
    learner_name: str | None = None
    notes: str | None = None


def foreign_number(phone: str, explanation: Lang) -> Identified | None:
    """Refuse a number from outside Spain before it is read back: Faro's learners are in Spain.

    The precheck of ``check_number``. Its reply says ``said``.
    """
    if phone.startswith("+34"):
        return None
    return Identified(
        status="foreign",
        said=curriculum.foreign_number(explanation),
        phone=phone,
        explanation=explanation,
    )


class Language(BaseModel):
    """What a prefetch of the caller's number returns, before anything is said."""

    explanation: Lang
    phone_spaced: str


class Exercise(BaseModel):
    """What ``continue_lesson`` and ``grade_answer`` return: code's line, and what waits."""

    said: str
    waiting: str | None = None
    helped: bool = False
    lesson_complete: bool = False


class Hinted(BaseModel):
    """What ``give_hint`` returns: the hint, said by code, and that help was given."""

    helped: bool
    said: str


class Said(BaseModel):
    """A line code says, and the learner's notes as they stand now."""

    said: str
    notes: str | None = None


def brief(learner: Learner) -> str:
    """Write what code knows about a learner, one line each. Context, never instructions."""
    lines = [f"Learner: {learner.name}. Explains in {learner.explanation}."]
    lines += [f"Remembered ({fact.key}): {fact.text}" for fact in learner.memory]
    done = [curriculum.lesson(done).topic["EN"] for done in learner.completed]
    lines.append(
        f"Plan: learning {learner.target} {learner.goal}, level {learner.level}, "
        f"{learner.minutes}-minute sessions."
    )
    lines.append(f"Lessons completed: {', '.join(done) or 'none yet'}.")
    return "\n".join(lines)


def created_line(learner: Learner) -> str:
    """Say a new learner's plan, in their language, and offer the first lesson."""
    language = curriculum.LANGUAGE_NAMES[learner.explanation][learner.target]
    purpose = curriculum.GOAL_NAMES[learner.explanation][learner.goal]
    topics = [lesson.topic[learner.explanation] for lesson in learner.lessons]
    if learner.explanation == "EN":
        outline = f"{', '.join(topics[:-1])} and {topics[-1]}"
        return (
            f"Nice to meet you, {learner.name}. Your plan is {language} {learner.level} "
            f"{purpose}: {outline}. Shall we start the first lesson now?"
        )
    outline = f"{', '.join(topics[:-1])} y {topics[-1]}"
    return (
        f"Encantado, {learner.name}. Tu plan es {language} {learner.level} {purpose}: "
        f"{outline}. ¿Empezamos ahora la primera lección?"
    )


@dataclass
class Desk:
    """The tools, on one store."""

    store: Store

    def _learner(self, learner_id: int, explanation: Lang) -> Learner:
        """Return a learner, with the language this call explains in."""
        learner = self.store.get(learner_id)
        if learner.explanation != explanation:
            learner = learner.model_copy(update={"explanation": explanation})
            self.store.save(learner)
        return learner

    def language_of(self, phone: str) -> Language:
        """Return the language to explain in for a caller's number, before the greeting.

        Nothing about the learner is said or shown: the number is not confirmed yet.
        """
        found = self.store.find(phone)
        digits = phone.removeprefix("+34")
        spaced = (
            f"+34 {digits[:3]} {digits[3:6]} {digits[6:]}" if phone.startswith("+34") else phone
        )
        return Language(explanation=found.explanation if found else "EN", phone_spaced=spaced)

    def _identify(self, phone: str, explanation: Lang) -> Identified:
        """Return who the caller is, and the line that opens their call."""
        learner = self.store.find(phone)
        if learner is None:
            return Identified(
                status="new",
                said=curriculum.signup_offer(explanation),
                phone=phone,
                explanation=explanation,
            )
        learner = self._learner(learner.id, explanation)
        if learner.finished:
            said = curriculum.finished_plan(learner.name, explanation)
        else:
            previous = learner.completed[-1] if learner.completed else ""
            said = curriculum.status(
                learner.name,
                explanation,
                learner.lessons[learner.lesson].id,
                learner.item,
                previous,
                False,
            )
        return Identified(
            status="known",
            said=said,
            phone=phone,
            explanation=explanation,
            learner_id=learner.id,
            learner_name=learner.name,
            notes=brief(learner),
        )

    def use_this_number(self, phone: str, explanation: Lang) -> Identified:
        """The caller said yes to the number read back in the greeting: look them up."""
        return self._identify(phone, explanation)

    def check_number(self, phone: NumberSaid, explanation: Lang) -> Identified:
        """Look the caller up by the number they said. It first reads the number back.

        Call it again with the same number after their yes.
        """
        return self._identify(phone, explanation)

    def create_profile(
        self,
        name: Annotated[
            str, Field(min_length=1, max_length=40, description="What they want to be called.")
        ],
        learn: Annotated[
            Literal["English", "Spanish"],
            Field(description="The language they want to learn."),
        ],
        goal: Annotated[
            Literal["travel", "work", "daily-life"],
            Field(description="What it is for: travel, work or daily life."),
        ],
        level: Annotated[
            Level,
            Field(description="Their level: A1 for a beginner, A2 for some, B1 for confident."),
        ],
        phone: str,
        explanation: Lang,
    ) -> Identified:
        """Create the new learner's profile and plan. It first reads it back.

        Call it again with the same values after their yes.
        """
        target: Lang = "EN" if learn == "English" else "ES"
        purpose: Goal = "daily" if goal == "daily-life" else goal
        if target == explanation:
            raise ToolError(
                "They would learn the language they already explain in. Ask which language "
                "they want to learn: English or Spanish."
            )
        try:
            learner = self.store.create(phone, name.strip(), explanation, target, purpose, level)
        except ValueError as error:
            raise ToolError(str(error)) from error
        return Identified(
            status="created",
            said=created_line(learner),
            phone=phone,
            explanation=explanation,
            learner_id=learner.id,
            learner_name=learner.name,
            notes=brief(learner),
        )

    def continue_lesson(self, learner_id: int, explanation: Lang) -> Exercise:
        """Put the next exercise of their plan to the learner. Code says it."""
        learner = self._learner(learner_id, explanation)
        if learner.finished:
            return Exercise(said=curriculum.finished_plan(learner.name, explanation))
        return Exercise(
            said=curriculum.exercise(explanation, learner.waiting), waiting=learner.waiting
        )

    def give_hint(
        self,
        hint: Annotated[
            str,
            Field(
                min_length=1,
                max_length=300,
                description=(
                    "What you say to help, in the explanation language: an answer to their "
                    "question, a slower explanation or a hint. Never the whole answer."
                ),
            ),
        ],
        waiting: str,
    ) -> Hinted:
        """Help with the exercise waiting. The application says your hint, so say nothing else.

        A right answer after a hint counts as practised with help.
        """
        return Hinted(helped=bool(waiting), said=hint.strip())

    def grade_answer(
        self,
        answer: Annotated[
            str,
            Field(
                min_length=1,
                max_length=600,
                description="The learner's attempt at the exercise, copied from what they said.",
            ),
        ],
        grade: Annotated[
            Grade,
            Field(
                description=(
                    "correct: does what was asked, in the target language, and can be "
                    "understood; small slips are fine. partly: in the target language, but "
                    "does only part of it or needs a real fix. wrong: does not do what was "
                    "asked, or is mostly not in the target language. unclear: too cut off "
                    "or garbled to judge. Judge meaning, never spelling or pronunciation."
                )
            ),
        ],
        feedback: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                description=(
                    "Said before the next line, in the explanation language. correct: brief "
                    "praise. partly or wrong: what was wrong, then the right sentence in the "
                    "target language. unclear: ask them to say it again."
                ),
            ),
        ],
        learner_id: int,
        explanation: Lang,
        waiting: str,
        helped: bool,
    ) -> Exercise:
        """Grade the learner's attempt at the exercise waiting. Code says what follows."""
        learner = self._learner(learner_id, explanation)
        if learner.waiting != waiting:
            # The exercise the model graded is not the one the store has waiting.
            raise ToolError("that exercise is no longer waiting: call continue_lesson")
        result = RESULTS[grade]
        if result == "independent" and helped:
            result = "assisted"
        attempts = [*learner.attempts, Attempt(item=waiting, answer=answer, result=result)]
        review = list(learner.review)
        if result in ("assisted", "not_yet") and waiting not in review:
            review.append(waiting)
        lesson = learner.lessons[learner.lesson]
        moved = result in ("independent", "assisted")
        update: dict[str, object] = {"attempts": attempts, "review": review}
        if moved and learner.item == len(lesson.items) - 1:
            update |= {
                "lesson": learner.lesson + 1,
                "item": 0,
                "completed": [*learner.completed, lesson.id],
            }
        elif moved:
            update |= {"item": learner.item + 1}
        learner = learner.model_copy(update=update)
        self.store.save(learner)
        if moved and learner.item == 0:
            following = (
                "" if learner.finished else learner.lessons[learner.lesson].topic[explanation]
            )
            done = curriculum.lesson_done(explanation, lesson.topic[explanation], following)
            return Exercise(said=f"{feedback} {done}", lesson_complete=True)
        again = not moved
        line = curriculum.exercise(explanation, learner.waiting, again=again)
        return Exercise(said=f"{feedback} {line}", waiting=learner.waiting)

    def read_progress(self, learner_id: int, explanation: Lang) -> Said:
        """Say what the learner has done, what needs review and what comes next. Changes nothing."""
        learner = self._learner(learner_id, explanation)
        done = [curriculum.lesson(done).topic[explanation] for done in learner.completed]
        review = [
            curriculum.item(item)[1].instruction[explanation].rstrip(".") for item in learner.review
        ]
        following = "" if learner.finished else learner.lessons[learner.lesson].topic[explanation]
        return Said(said=curriculum.recap(explanation, done, review, following))

    def remember(
        self,
        key: Annotated[
            str,
            Field(min_length=1, max_length=40, description="A short stable name, like hobby."),
        ],
        text: Annotated[
            str,
            Field(
                min_length=1,
                max_length=200,
                description='What they told you, in one short sentence, like "Enjoys climbing."',
            ),
        ],
        learner_id: int,
        explanation: Lang,
    ) -> Said:
        """Remember something the learner told you about themselves, for later calls.

        A correction uses the same key as the fact it replaces.
        """
        learner = self._learner(learner_id, explanation)
        key = key.strip().lower()
        memory = [
            *(fact for fact in learner.memory if fact.key != key),
            Fact(key=key, text=text.strip()),
        ]
        learner = learner.model_copy(update={"memory": memory[-20:]})
        self.store.save(learner)
        return Said(said=curriculum.noted(explanation), notes=brief(learner))
