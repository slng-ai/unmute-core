"""Faro's store and tools, offline: no model, no network, a store in memory.

The whole calls run live, on both frameworks, from ``../scenarios`` (see the README).
"""

import pytest
from unmute_core.state import call_info
from unmute_core.tools import ToolError

from faro import curriculum
from faro.agent import GREETING, Faro
from faro.desk import Desk
from faro.store import Store

LUCIA, EMMA, NEW = "+34600111222", "+34600555666", "+34600777888"


def desk() -> Desk:
    return Desk(Store(":memory:"))


def emma(faro_desk: Desk) -> int:
    found = faro_desk.use_this_number(EMMA, "EN")
    assert found.learner_id is not None
    return found.learner_id


async def test_a_known_caller_is_greeted_in_their_own_language_and_named_only_after_a_yes() -> None:
    faro = Faro(Store(":memory:"))
    state = await faro.new_call(call_info("inbound", remote_number=LUCIA, agent_id="faro"))
    assert GREETING.line(state) == (
        "Hi, I'm Faro, your language companion. Si prefieres hablar en español, dímelo. "
        "A ver si lo tengo bien: ¿tu número es +34 600 111 222?"
    )
    assert faro.agent(state).name == "faro_es"
    assert "Lucía" not in str(state.snapshot())


async def test_with_no_caller_id_the_model_asks_for_the_number() -> None:
    state = await Faro(Store(":memory:")).new_call()
    assert GREETING.line(state) is None
    assert "starting with plus thirty-four" in GREETING.instruction


def test_a_returning_learner_hears_where_they_stopped() -> None:
    found = desk().use_this_number(LUCIA, "ES")
    assert found.said == (
        "Hola, Lucía. Estabas practicando cómo pedir en una cafetería. Nos quedamos en el "
        "segundo ejercicio. ¿Seguimos o prefieres repasar primero?"
    )
    assert found.notes is not None and "Enjoys hiking." in found.notes


def test_an_unknown_number_is_offered_a_profile_and_learns_nothing() -> None:
    found = desk().use_this_number(NEW, "EN")
    assert found.status == "new" and found.learner_id is None
    assert found.said.startswith("I can't find a profile with that number.")


def test_only_a_spanish_number_is_taken() -> None:
    with pytest.raises(ValueError, match="not a Spanish number"):
        from faro.desk import spanish

        spanish("+33612345678")


def test_a_lesson_moves_only_on_graded_answers() -> None:
    faro_desk = desk()
    learner = emma(faro_desk)
    first = faro_desk.continue_lesson(learner, "EN")
    assert (
        first.said
        == "Shopping, first exercise. Answer in Spanish: Ask for a kilo of apples at the market."
    )
    assert first.waiting == "es-daily-2-1"
    wrong = faro_desk.grade_answer(
        "manzana", "wrong", "Almost.", learner, "EN", "es-daily-2-1", False
    )
    assert wrong.said == "Almost. Try again, in Spanish: Ask for a kilo of apples at the market."
    assert wrong.waiting == "es-daily-2-1"
    right = faro_desk.grade_answer(
        "Un kilo de manzanas.", "correct", "Great!", learner, "EN", "es-daily-2-1", False
    )
    assert (
        right.said == "Great! Shopping, second exercise. Answer in Spanish: Ask how much it costs."
    )
    stored = faro_desk.store.get(learner)
    assert [a.result for a in stored.attempts] == ["not_yet", "independent"]
    assert stored.review == ["es-daily-2-1"]


def test_a_right_answer_after_a_hint_counts_as_help() -> None:
    faro_desk = desk()
    learner = emma(faro_desk)
    faro_desk.continue_lesson(learner, "EN")
    assert faro_desk.give_hint("Apples is manzanas.", "es-daily-2-1").helped
    faro_desk.grade_answer(
        "Un kilo de manzanas.", "correct", "Great!", learner, "EN", "es-daily-2-1", True
    )
    assert faro_desk.store.get(learner).attempts[-1].result == "assisted"


def test_the_check_completes_the_lesson() -> None:
    faro_desk = desk()
    learner = emma(faro_desk)
    for item in ("es-daily-2-1", "es-daily-2-2"):
        faro_desk.grade_answer("ok", "correct", "Good.", learner, "EN", item, False)
    done = faro_desk.grade_answer("ok", "correct", "Perfect!", learner, "EN", "es-daily-2-3", False)
    assert done.lesson_complete and done.waiting is None
    assert done.said == (
        "Perfect! Lesson complete: shopping. Next is talking about your routine. Shall we go "
        "on, or stop here for today?"
    )
    assert faro_desk.store.get(learner).completed == ["es-daily-1", "es-daily-2"]


def test_grading_an_exercise_that_is_not_waiting_is_refused() -> None:
    faro_desk = desk()
    learner = emma(faro_desk)
    with pytest.raises(ToolError, match="no longer waiting"):
        faro_desk.grade_answer("ok", "correct", "Good.", learner, "EN", "es-daily-3-1", False)


def test_a_new_profile_is_read_back_then_made() -> None:
    faro_desk = desk()
    made = faro_desk.create_profile("Nico", "Spanish", "travel", "A1", NEW, "EN")
    assert made.said.startswith("Nice to meet you, Nico. Your plan is Spanish A1 for travel:")
    assert faro_desk.store.count() == 4
    with pytest.raises(ToolError, match="already exists"):
        faro_desk.create_profile("Nico", "Spanish", "travel", "A1", NEW, "EN")
    with pytest.raises(ToolError, match="they already explain in"):
        faro_desk.create_profile("Ana", "English", "work", "A2", "+34600999000", "EN")


def test_memory_corrections_replace_the_old_fact() -> None:
    faro_desk = desk()
    learner = emma(faro_desk)
    noted = faro_desk.remember("Prompts", "Likes long prompts.", learner, "EN")
    assert noted.said == "Noted."
    assert [f.text for f in faro_desk.store.get(learner).memory] == ["Likes long prompts."]


def test_a_recap_changes_nothing() -> None:
    faro_desk = desk()
    diego = faro_desk.use_this_number("+34600333444", "ES").learner_id
    assert diego is not None
    before = faro_desk.store.get(diego)
    recap = faro_desk.read_progress(diego, "ES")
    assert recap.said.startswith(
        "Has completado: presentarte en el trabajo, hablar de lo que hiciste."
    )
    assert faro_desk.store.get(diego) == before


def test_the_curriculum_names_every_lesson_in_both_languages() -> None:
    for plan in curriculum.LESSONS.values():
        assert len(plan) == 4
        for lesson in plan:
            assert set(lesson.topic) == {"EN", "ES"}
            assert lesson.items[-1].check
