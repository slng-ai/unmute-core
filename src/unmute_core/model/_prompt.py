"""How a prompt's ``{{name}}`` placeholders are found and filled.

The rule is the SLNG Context Router's, so a prompt filled here reads exactly as the router
fills it: ``{{name}}`` or ``{{ name }}``, one pass, and only in system and developer
messages. A path into a shape is ``{{customer.status}}`` or ``{{customer__status}}``. The
router reads only the second form, so ``router_messages`` rewrites the first before a
request goes to it, and every variable is keyed by that form.
"""

from collections.abc import Callable, Iterator
from typing import Any

from unmute_core.state import CallState
from unmute_core.state._paths import PLACEHOLDER, ValuePathError, parse_path

FILLED_ROLES = frozenset({"system", "developer"})


class MissingVariablesError(ValueError):
    """A prompt names placeholders that have no value.

    Attributes:
        names: The missing names, sorted.
    """

    def __init__(self, names: list[str]) -> None:
        """Keep the sorted names.

        Args:
            names: The missing placeholder names.
        """
        self.names = sorted(set(names))
        super().__init__(f"Missing required template variables: {', '.join(self.names)}")


def _is_text_part(part: object) -> bool:
    """Return True for a content part the router fills: ``{"type": "text", "text": str}``."""
    return (
        isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
    )


def _texts(message: dict[str, Any]) -> Iterator[str]:
    """Yield the text a fill reads in one message: system and developer text only."""
    if message.get("role") not in FILLED_ROLES:
        return
    content = message.get("content")
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        yield from (part["text"] for part in content if _is_text_part(part))


def key(name: str) -> str:
    """Return the router's name for a placeholder: ``customer.status`` is ``customer__status``."""
    return name.replace(".", "__")


def check_keys(text: str, where: str, limit: int) -> None:
    """Refuse a placeholder whose key is longer than a model call takes.

    Only a model call that fills the prompt on its own side, like the router, has a limit.
    See ``variable_key_max``.

    Args:
        text: A prompt.
        where: What the text is, for the message, like ``task 'greet'``.
        limit: The longest key the model call takes.

    Raises:
        ValueError: Naming the placeholder and its key.
    """
    for match in PLACEHOLDER.finditer(text):
        if len(name := key(match[1])) > limit:
            raise ValueError(
                f"{where}: the placeholder {{{{{match[1]}}}}} becomes the key {name!r}, longer "
                f"than the {limit} characters the model call takes. Use a shorter name."
            )


def placeholders(messages: list[dict[str, Any]]) -> list[str]:
    """Return the placeholder names the messages use, as written, in first-seen order."""
    names = (
        match[1]
        for message in messages
        for text in _texts(message)
        for match in PLACEHOLDER.finditer(text)
    )
    return list(dict.fromkeys(names))


def _map_texts(message: dict[str, Any], change: Callable[[str], str]) -> dict[str, Any]:
    """Return a copy of one message with ``change`` applied to its system or developer text."""
    if message.get("role") not in FILLED_ROLES:
        return message
    content = message.get("content")
    if isinstance(content, str):
        return {**message, "content": change(content)}
    if isinstance(content, list):
        parts = [
            {**part, "text": change(part["text"])} if _is_text_part(part) else part
            for part in content
        ]
        return {**message, "content": parts}
    return message


def _fill_text(text: str, variables: dict[str, str]) -> str:
    """Replace each placeholder with its value, in one pass."""
    return PLACEHOLDER.sub(lambda match: variables[key(match[1])], text)


def router_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite each ``{{a.b}}`` to ``{{a__b}}``, the only form the router fills.

    Args:
        messages: The chat in OpenAI chat format.

    Returns:
        A new list. Messages the router does not fill are the same objects.
    """

    def rewrite(text: str) -> str:
        """Rewrite the dotted placeholders of one text."""
        return PLACEHOLDER.sub(lambda match: "{{" + key(match[1]) + "}}", text)

    return [_map_texts(message, rewrite) for message in messages]


def folded_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold every system and developer message into one system message at the front.

    Some endpoints, like Gemini's OpenAI-compatible one, take only one system message. The
    texts join in order with a blank line. The other messages keep their order. A chat with
    one system message and no developer message comes back unchanged.

    Args:
        messages: The chat in OpenAI chat format.

    Returns:
        A new list.
    """
    system = [message for message in messages if message.get("role") in FILLED_ROLES]
    if not system or (len(system) == 1 and system[0]["role"] == "system"):
        return list(messages)
    text = "\n\n".join(text for message in system for text in _texts(message))
    rest = [message for message in messages if message.get("role") not in FILLED_ROLES]
    return [{"role": "system", "content": text}, *rest]


def fill_messages(
    messages: list[dict[str, Any]], variables: dict[str, str]
) -> list[dict[str, Any]]:
    """Fill the placeholders of the system and developer messages, as the router does.

    User and assistant messages are never changed. A value that holds ``{{x}}`` stays as it
    is. The input list is not changed.

    Args:
        messages: The chat in OpenAI chat format.
        variables: The router's placeholder name (``key``) to its text.

    Returns:
        A new list with the filled messages.

    Raises:
        MissingVariablesError: If a placeholder has no value. The router refuses it with a 422.
    """
    if missing := [name for name in placeholders(messages) if key(name) not in variables]:
        raise MissingVariablesError(missing)
    return [_map_texts(message, lambda text: _fill_text(text, variables)) for message in messages]


def _render(name: str, state: CallState, site: str) -> str:
    """Render one placeholder from the state.

    Raises:
        KeyError: If it names no value of the state.
    """
    try:
        path = parse_path(name, state.name_set)
    except ValuePathError:
        raise KeyError(name) from None
    return state.render(path.text, site)


def template_variables(
    messages: list[dict[str, Any]],
    state: CallState | None,
    site: str = "",
    *,
    as_written: bool = False,
) -> dict[str, str]:
    """Render every placeholder the messages use from the call state.

    Args:
        messages: The chat in OpenAI chat format.
        state: The call state, or None for an agent without one.
        site: The prompt reading the values, like ``task:verify_customer``. A value waiting
            for the caller to confirm it reads as empty, except at its own task.
        as_written: True for prompts that are not templates: a ``{{name}}`` in them is
            plain text, so nothing is read and the result is empty.

    Returns:
        The router's placeholder name (``key``) to its text, ready for ``fill_messages``
        or the router.

    Raises:
        MissingVariablesError: If a placeholder names no value of the state.
    """
    names = [] if as_written else placeholders(messages)
    if not names:
        return {}
    if state is None:
        raise MissingVariablesError(names)
    variables = {}
    missing = []
    for name in names:
        try:
            variables[key(name)] = _render(name, state, site)
        except KeyError:
            missing.append(name)
    if missing:
        raise MissingVariablesError(missing)
    return variables


def render_prompt(text: str, state: CallState | None, site: str = "") -> str:
    """Fill a prompt's placeholders from the call state, as the model will read it.

    Use it to give a framework's own LLM a filled prompt, with no ``ModelCall`` in between.
    Render it again after a save, so the prompt shows the new values.

    Args:
        text: The prompt, like ``"Help {{customer.name}} with their order."``.
        state: The call state, or None for an agent without one.
        site: The prompt reading the values, like ``task:verify_customer``.

    Returns:
        The prompt with every placeholder filled.

    Raises:
        MissingVariablesError: If a placeholder names no value of the state.
    """
    messages = [{"role": "system", "content": text}]
    return fill_messages(messages, template_variables(messages, state, site))[0]["content"]
