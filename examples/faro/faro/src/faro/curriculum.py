"""Faro's authored material: the lessons, and every line code says to the learner.

Ported from unmute-beat's ``faro-language-learning`` (``faro/curriculum.py``), without its
placement bank. Everything a lesson must repeat is here, in code: which items a lesson
has and in what order, which item is the lesson's check, and the words of every question
put to the learner. The model grades an answer in typed fields; it never picks an item.

Every instruction is written twice, in English and in Spanish. The learner hears it in
their explanation language and answers in the language they are learning. The lessons are
the same for both target languages, so each is written once per goal and given an id per
target: ``en-travel-2`` is the café lesson for someone learning English, ``es-travel-2``
the same lesson for someone learning Spanish.
"""

from dataclasses import dataclass
from typing import Literal

type Lang = Literal["EN", "ES"]
type Goal = Literal["travel", "work", "daily"]
type Result = Literal["independent", "assisted", "not_yet", "insufficient_evidence"]
type Level = Literal["A1", "A2", "B1", "B2"]


LANGUAGE_NAMES: dict[Lang, dict[Lang, str]] = {
    "ES": {"EN": "inglés", "ES": "español"},
    "EN": {"EN": "English", "ES": "Spanish"},
}
"""A target language's name, said in the explanation language."""


def target_of(record_id: str) -> Lang:
    """Return the language a lesson or item id teaches: ``en-travel-2-1`` teaches English."""
    return "EN" if record_id.startswith("en-") else "ES"


@dataclass(frozen=True)
class Item:
    """One practice item: what to do, in each explanation language. `check` marks the lesson's independent check."""

    id: str
    instruction: dict[Lang, str]
    check: bool = False


@dataclass(frozen=True)
class Lesson:
    """One lesson: its topic said in each explanation language, its can-do objective, and its items."""

    id: str
    topic: dict[Lang, str]
    objective: str
    items: tuple[Item, ...]


@dataclass(frozen=True)
class _Plan:
    """One lesson as written: topic in each language, objective, and (English, Spanish) instructions."""

    es: str
    en: str
    objective: str
    items: tuple[tuple[str, str], ...]


_GOALS: dict[Goal, tuple[_Plan, ...]] = {
    "travel": (
        _Plan(
            "presentarte",
            "introducing yourself",
            "I can introduce myself and say where I am from.",
            (
                ("Say hello and tell me your name.", "Saluda y dime cómo te llamas."),
                (
                    "Tell me where you are from and where you live now.",
                    "Dime de dónde eres y dónde vives ahora.",
                ),
                (
                    "Introduce yourself to a new friend at a hostel: your name, your city and one hobby.",
                    "Preséntate a alguien en un albergue: tu nombre, tu ciudad y una afición.",
                ),
            ),
        ),
        _Plan(
            "pedir en una cafetería",
            "ordering at a café",
            "I can order food and drink politely and ask for the total.",
            (
                (
                    "You are at a café. Ask politely for a drink.",
                    "Estás en una cafetería. Pide una bebida con educación.",
                ),
                (
                    "Now say how many you want: ask for two of something.",
                    "Ahora di cuántos quieres: pide dos de algo.",
                ),
                (
                    "Order two drinks and ask how much it is in total.",
                    "Pide dos bebidas y pregunta cuánto es en total.",
                ),
            ),
        ),
        _Plan(
            "resolver un problema en el hotel",
            "solving a problem at the hotel",
            "I can explain a simple problem and ask for help.",
            (
                (
                    "Tell the receptionist that your room is too cold.",
                    "Dile al recepcionista que tu habitación está muy fría.",
                ),
                (
                    "Ask if you can change to another room.",
                    "Pregunta si puedes cambiar de habitación.",
                ),
                (
                    "Your shower does not work. Explain the problem and ask when it will be fixed.",
                    "La ducha no funciona. Explica el problema y pregunta cuándo la arreglan.",
                ),
            ),
        ),
        _Plan(
            "practicar una conversación completa",
            "practising a full conversation",
            "I can handle a short travel conversation from start to finish.",
            (
                (
                    "Ask a stranger for directions to the train station.",
                    "Pregunta a alguien cómo llegar a la estación de tren.",
                ),
                (
                    "Buy a return ticket to the airport.",
                    "Compra un billete de ida y vuelta al aeropuerto.",
                ),
                (
                    "Check in at a hotel: give your name, ask about breakfast and say thank you.",
                    "Haz el registro en un hotel: di tu nombre, pregunta por el desayuno y da las gracias.",
                ),
            ),
        ),
    ),
    "work": (
        _Plan(
            "presentarte en el trabajo",
            "introducing yourself at work",
            "I can introduce myself and my role at work.",
            (
                ("Tell me your name and your job.", "Dime tu nombre y tu trabajo."),
                (
                    "Explain what you do on a normal work day.",
                    "Explica qué haces en un día normal de trabajo.",
                ),
                (
                    "Introduce yourself to a new colleague: your name, your role and your team.",
                    "Preséntate a un compañero nuevo: tu nombre, tu puesto y tu equipo.",
                ),
            ),
        ),
        _Plan(
            "hablar de lo que hiciste",
            "talking about what you did",
            "I can talk about completed work in the past tense.",
            (
                (
                    "Tell me three things you did yesterday at work.",
                    "Dime tres cosas que hiciste ayer en el trabajo.",
                ),
                (
                    "Tell me about a problem you solved last week.",
                    "Cuéntame un problema que resolviste la semana pasada.",
                ),
                (
                    "Give a short update: what you finished this week and what went wrong.",
                    "Da una actualización corta: qué terminaste esta semana y qué salió mal.",
                ),
            ),
        ),
        _Plan(
            "describir un proyecto",
            "describing a project",
            "I can describe a project: its goal, my part and its status.",
            (
                (
                    "Say what the goal of a project you work on is.",
                    "Di cuál es el objetivo de un proyecto en el que trabajas.",
                ),
                ("Say what your part in that project is.", "Di cuál es tu parte en ese proyecto."),
                (
                    "Describe the project to a new manager: its goal, your role and the next step.",
                    "Describe el proyecto a una jefa nueva: su objetivo, tu papel y el siguiente paso.",
                ),
            ),
        ),
        _Plan(
            "participar en una reunión",
            "taking part in a meeting",
            "I can give an opinion and agree or disagree politely in a meeting.",
            (
                (
                    "Say that you agree with an idea, and why.",
                    "Di que estás de acuerdo con una idea y por qué.",
                ),
                (
                    "Disagree politely with a suggestion.",
                    "Muestra tu desacuerdo con una propuesta, con educación.",
                ),
                (
                    "In a meeting, suggest a new deadline and give one reason.",
                    "En una reunión, propón una nueva fecha de entrega y da una razón.",
                ),
            ),
        ),
    ),
    "daily": (
        _Plan(
            "presentarte",
            "introducing yourself",
            "I can introduce myself and ask simple questions.",
            (
                ("Say hello and tell me your name.", "Saluda y dime cómo te llamas."),
                ("Ask me my name and where I live.", "Pregúntame cómo me llamo y dónde vivo."),
                (
                    "Introduce yourself to a neighbour and ask one question.",
                    "Preséntate a un vecino y hazle una pregunta.",
                ),
            ),
        ),
        _Plan(
            "ir de compras",
            "shopping",
            "I can ask for things in a shop and ask the price.",
            (
                (
                    "Ask for a kilo of apples at the market.",
                    "Pide un kilo de manzanas en el mercado.",
                ),
                ("Ask how much it costs.", "Pregunta cuánto cuesta."),
                (
                    "Buy bread and milk and ask how much it is in total.",
                    "Compra pan y leche y pregunta cuánto es en total.",
                ),
            ),
        ),
        _Plan(
            "hablar de tu rutina",
            "talking about your routine",
            "I can describe my daily routine.",
            (
                ("Say what time you get up.", "Di a qué hora te levantas."),
                ("Say what you do in the evening.", "Di qué haces por la tarde."),
                (
                    "Describe a normal Saturday, from morning to night.",
                    "Describe un sábado normal, de la mañana a la noche.",
                ),
            ),
        ),
        _Plan(
            "hacer planes",
            "making plans",
            "I can make and accept plans with a friend.",
            (
                ("Invite a friend to the cinema.", "Invita a un amigo al cine."),
                ("Suggest a time and a place to meet.", "Propón una hora y un lugar para quedar."),
                (
                    "Make a plan for Sunday with a friend: what, when and where.",
                    "Haz un plan para el domingo con un amigo: qué, cuándo y dónde.",
                ),
            ),
        ),
    ),
}
"""Four lessons per goal, written once. The last item of each is the independent check."""


def _built(target: Lang, goal: Goal, number: int, plan: _Plan) -> Lesson:
    lesson_id = f"{target.lower()}-{goal}-{number}"
    items = tuple(
        Item(f"{lesson_id}-{n}", {"EN": en, "ES": es}, check=n == len(plan.items))
        for n, (en, es) in enumerate(plan.items, start=1)
    )
    return Lesson(lesson_id, {"ES": plan.es, "EN": plan.en}, plan.objective, items)


LESSONS: dict[tuple[Lang, Goal], tuple[Lesson, ...]] = {
    (target, goal): tuple(_built(target, goal, n, plan) for n, plan in enumerate(plans, start=1))
    for target in ("EN", "ES")
    for goal, plans in _GOALS.items()
}
"""Every plan there is: four lessons per target language and goal, in this order."""


def lessons(target: Lang, goal: Goal) -> tuple[Lesson, ...]:
    """Return the four lessons of one plan, in order."""
    return LESSONS[target, goal]


def lesson(lesson_id: str) -> Lesson:
    """The lesson with this id, from any plan."""
    for plan in LESSONS.values():
        for found in plan:
            if found.id == lesson_id:
                return found
    raise KeyError(lesson_id)


def item(item_id: str) -> tuple[Lesson, Item]:
    """The item with this id and the lesson it belongs to."""
    lesson_id, _, _ = item_id.rpartition("-")
    found = lesson(lesson_id)
    return found, next(i for i in found.items if i.id == item_id)


GOAL_NAMES: dict[Lang, dict[Goal, str]] = {
    "ES": {"travel": "para viajar", "work": "para el trabajo", "daily": "para el día a día"},
    "EN": {"travel": "for travel", "work": "for work", "daily": "for daily life"},
}

NUMBERS: dict[Lang, dict[int, str]] = {
    "ES": {1: "primer", 2: "segundo", 3: "tercer"},
    "EN": {1: "first", 2: "second", 3: "third"},
}


def spaced(phone: str) -> str:
    """A Spanish number the way it is read out: +34 600 111 222."""
    digits = phone.removeprefix("+34")
    return f"+34 {digits[:3]} {digits[3:6]} {digits[6:]}"


GREETING = "Hi, I'm Faro, your language companion. Si prefieres hablar en español, dímelo."
"""The opening words, in English, offering Spanish. The identify step's read-back or question follows them."""


def read_back(phone: str, explanation: Lang) -> str:
    """Say a number back for a yes."""
    if explanation == "EN":
        return f"Let me check: is your number {spaced(phone)}?"
    return f"A ver si lo tengo bien: ¿tu número es {spaced(phone)}?"


def foreign_number(explanation: Lang) -> str:
    """Ask again for a number from outside Spain."""
    if explanation == "EN":
        return "Faro only takes Spanish numbers, starting with plus thirty-four. What is your Spanish number?"
    return "Faro solo acepta números españoles, que empiezan por más treinta y cuatro. ¿Cuál es tu número español?"


def signup_offer(explanation: Lang) -> str:
    """Offer a profile to a number that has none."""
    if explanation == "EN":
        return "I can't find a profile with that number. Would you like to create one and prepare a learning plan?"
    return "No encuentro un perfil con ese número. ¿Quieres crear uno y preparar un plan de aprendizaje?"


def status(
    name: str, explanation: Lang, lesson_id: str, item_index: int, previous: str, asked: bool
) -> str:
    """The returning learner's opening: where they stopped, and the two choices every return offers.

    `item_index` is the exercise waiting, and `asked` whether it was already
    put to them: either means the lesson is under way. `previous` is the
    lesson finished before this one, or ''.
    """
    topic = lesson(lesson_id).topic[explanation]
    ordinal = NUMBERS[explanation].get(item_index + 1, str(item_index + 1))
    before = lesson(previous).topic[explanation] if previous else ""
    if explanation == "EN":
        if item_index or asked:
            where = f"You were practising {topic}. We stopped at the {ordinal} exercise."
        elif before:
            where = f"You finished {before}. The next lesson, {topic}, is ready to start."
        else:
            where = f"Your first lesson, {topic}, is ready to start."
        return f"Hi, {name}. {where} Shall we carry on, or would you like a recap first?"
    if item_index or asked:
        where = f"Estabas practicando cómo {topic}. Nos quedamos en el {ordinal} ejercicio."
    elif before:
        where = f"Terminaste la lección de {before}. La siguiente, {topic}, está lista."
    else:
        where = f"Tu primera lección, {topic}, está lista."
    return f"Hola, {name}. {where} ¿Seguimos o prefieres repasar primero?"


def plan_proposal(target: Lang, goal: Goal, level: Level, minutes: int, explanation: Lang) -> str:
    """The exact words of a plan put to the learner for agreement: goal, level, the four lessons, the length."""
    topics = [plan_lesson.topic[explanation] for plan_lesson in lessons(target, goal)]
    language = LANGUAGE_NAMES[explanation][target]
    purpose = GOAL_NAMES[explanation][goal]
    if explanation == "EN":
        outline = f"{', '.join(topics[:-1])} and {topics[-1]}"
        return (
            f"We will start with {language} {level} {purpose}: {outline}. "
            f"Sessions will be about {minutes} minutes. Does that sound good?"
        )
    outline = f"{', '.join(topics[:-1])} y {topics[-1]}"
    return (
        f"Empezaremos con {language} {level} {purpose}: {outline}. "
        f"Serán sesiones de unos {minutes} minutos. ¿Te parece bien?"
    )


def plan_active(explanation: Lang) -> str:
    """Say that a plan is now active, and offer the first lesson."""
    if explanation == "EN":
        return "Done: your plan is active. Shall we start the first lesson now?"
    return "Hecho: tu plan ya está activo. ¿Empezamos ahora la primera lección?"


def exercise(explanation: Lang, item_id: str, *, again: bool = False) -> str:
    """An exercise as the learner hears it: in their own language, saying which language to answer in.

    The first time with where they are in the lesson; asked again, only the
    short retry, so a second try never repeats the whole introduction.
    """
    found, waiting = item(item_id)
    language = LANGUAGE_NAMES[explanation][target_of(item_id)]
    instruction = waiting.instruction[explanation]
    if again:
        return (
            f"Try again, in {language}: {instruction}"
            if explanation == "EN"
            else f"Otra vez, en {language}: {instruction}"
        )
    ordinal = NUMBERS[explanation].get(found.items.index(waiting) + 1, "")
    topic = found.topic[explanation].capitalize()
    if explanation == "EN":
        return f"{topic}, {ordinal} exercise. Answer in {language}: {instruction}"
    return f"{topic}, {ordinal} ejercicio. Responde en {language}: {instruction}"


def lesson_done(explanation: Lang, finished: str, following: str) -> str:
    """Say that a lesson is complete, and what comes next."""
    if explanation == "EN":
        if not following:
            return f"Lesson complete: {finished}. That was the last lesson of your plan. Well done!"
        return f"Lesson complete: {finished}. Next is {following}. Shall we go on, or stop here for today?"
    if not following:
        return f"Lección completada: {finished}. Era la última lección de tu plan. ¡Muy bien!"
    return f"Lección completada: {finished}. La siguiente es {following}. ¿Seguimos o lo dejamos por hoy?"


def next_step(explanation: Lang, item_id: str) -> str:
    """Where the next session starts, named by lesson and exercise number rather than by the prompt itself."""
    found, waiting = item(item_id)
    ordinal = NUMBERS[explanation].get(found.items.index(waiting) + 1, "")
    topic = found.topic[explanation]
    if explanation == "EN":
        return f"Next time: the {ordinal} exercise of {topic}."
    return f"La próxima vez: el {ordinal} ejercicio de {topic}."


def summary(explanation: Lang, learned: int, next_item: str) -> str:
    """What a session did, written by code from the store's count, for the plan's record. No model writes it."""
    if explanation == "ES":
        said = (
            f"Hoy has practicado {learned} ejercicio{'s' if learned != 1 else ''}."
            if learned
            else "Hoy hemos hablado sin hacer ejercicios nuevos."
        )
    else:
        said = (
            f"Today you practised {learned} exercise{'s' if learned != 1 else ''}."
            if learned
            else "Today we talked without new exercises."
        )
    return f"{said} {next_step(explanation, next_item)}" if next_item else said


def finished_plan(name: str, explanation: Lang) -> str:
    """A returning learner whose plan has no lesson left: the two things they can do."""
    if explanation == "ES":
        return f"Hola, {name}. Has completado todas las lecciones de tu plan. ¿Hacemos una evaluación nueva o repasamos?"
    return f"Hi, {name}. You have finished every lesson in your plan. Shall we reassess, or recap?"


def hello(name: str, explanation: Lang) -> str:
    """Say hello by name."""
    return f"Hola, {name}." if explanation == "ES" else f"Hi, {name}."


def noted(explanation: Lang) -> str:
    """Say that something was noted."""
    return "Anotado." if explanation == "ES" else "Noted."


def recap(explanation: Lang, done: list[str], review: list[str], following: str) -> str:
    """A recap of the plan as the store has it: what is done, what needs review, what comes next. Completes nothing."""
    if explanation == "ES":
        parts = [
            f"Has completado: {', '.join(done)}."
            if done
            else "Todavía no has completado ninguna lección.",
            f"Para repasar: {'; '.join(review)}."
            if review
            else "No tienes nada pendiente de repaso.",
            f"Lo siguiente: {following}." if following else "Has terminado tu plan.",
        ]
        return " ".join([*parts, "¿Seguimos?"])
    parts = [
        f"You have completed: {', '.join(done)}."
        if done
        else "You have not completed a lesson yet.",
        f"To review: {'; '.join(review)}." if review else "Nothing is waiting for review.",
        f"Next: {following}." if following else "You have finished your plan.",
    ]
    return " ".join([*parts, "Shall we carry on?"])
