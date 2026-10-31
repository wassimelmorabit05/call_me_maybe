"""Generate schema-valid function calls with constrained decoding."""

from collections.abc import Callable
import math
from typing import Any

from .decoding import get_valid_next_tokens, mask_logits, select_next_token
from .input_loader import FunctionDefinition, Prompt
from .models import FunctionCallResult

MAX_VALUE_TOKENS = 64


def build_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Describe the available functions and the user's request."""
    lines = [
        "Choose the best function for this request and provide its arguments.",
        "",
        "Available functions:",
    ]
    for function in functions:
        parameters = ", ".join(
            f"{name} ({parameter.type})"
            for name, parameter in function.parameters.items()
        )
        lines.append(
            f"- {function.name}({parameters}): {function.description}"
        )
    lines.extend(("", f'User request: "{user_prompt}"'))
    return "\n".join(lines)


def _generate_value(
    text_so_far: str,
    model: Any,
    token_text_cache: dict[int, str],
    valid_next: Callable[[str], set[int]],
    can_stop: Callable[[str], bool],
    stop_text: str,
    kind: str,
    visualize: bool,
) -> tuple[str, str]:
    """Generate one constrained value and leave its boundary for the caller."""
    generated = ""

    for _ in range(MAX_VALUE_TOKENS):
        if kind == "string" and can_stop(generated):
            repetition = _repeating_tail(generated)
            if repetition:
                period, count = repetition
                removed = period * (count - 1)
                generated = generated[:-removed]
                text_so_far = text_so_far[:-removed]
                if visualize:
                    print(
                        f"    [string] stopped a repeating loop: {generated!r}"
                    )
                return generated, text_so_far

        valid_ids = valid_next(generated)
        stop_ids = set()
        if can_stop(generated):
            stop_ids = {
                token_id
                for token_id, text in token_text_cache.items()
                if text == stop_text
            }
        candidates = valid_ids | stop_ids
        if not candidates:
            raise ValueError(f"No valid continuation while generating {kind}")

        input_ids = model.encode(text_so_far).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)
        next_id = select_next_token(mask_logits(logits, candidates))
        stopped = next_id in stop_ids and next_id not in valid_ids

        if visualize:
            action = "STOP" if stopped else "PICK"
            print(f"    [{kind}] {action} -> {token_text_cache[next_id]!r}")

        if stopped:
            return generated, text_so_far

        addition = token_text_cache[next_id]
        generated += addition
        text_so_far += addition

    raise ValueError(
        f"{kind.capitalize()} did not finish within {MAX_VALUE_TOKENS} tokens"
    )


def _is_number_prefix(text: str, integer: bool) -> bool:
    if text in ("", "-"):
        return True
    if text.startswith("-"):
        text = text[1:]
    if text.count(".") > 1 or (integer and "." in text):
        return False
    return all(
        char.isascii() and (char.isdigit() or char == ".") for char in text
    )


def _is_complete_number(text: str) -> bool:
    if text in ("", "-") or text.endswith("."):
        return False
    try:
        return math.isfinite(float(text))
    except ValueError:
        return False


def generate_numeric_value(
    text_so_far: str,
    model: Any,
    token_text_cache: dict[int, str],
    stop_boundary_text: str,
    is_integer: bool,
    visualize: bool = False,
) -> tuple[float | int, str]:
    """Generate a number or integer from valid numeric tokens."""
    generated, text_so_far = _generate_value(
        text_so_far,
        model,
        token_text_cache,
        lambda current: {
            token_id
            for token_id, text in token_text_cache.items()
            if _is_number_prefix(current + text, is_integer)
        },
        _is_complete_number,
        stop_boundary_text,
        "number",
        visualize,
    )
    value = int(generated) if is_integer else float(generated)
    return value, text_so_far


def _pending_escape(text: str) -> bool:
    return (len(text) - len(text.rstrip("\\"))) % 2 == 1


def _valid_string_addition(current: str, addition: str) -> bool:
    for char in addition:
        if _pending_escape(current):
            if char not in ('"', "\\"):
                return False
        elif char == '"':
            return False
        current += char
    return True


def _repeating_tail(text: str) -> tuple[int, int] | None:
    for period in range(1, min(20, len(text) // 3) + 1):
        unit = text[-period:]
        count = 1
        start = len(text) - period
        while start >= period and text[start - period:start] == unit:
            count += 1
            start -= period
        if count >= 3:
            return period, count
    return None


def _unescape_string(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            index += 1
        result.append(text[index])
        index += 1
    return "".join(result)


def generate_string_value(
    text_so_far: str,
    model: Any,
    token_text_cache: dict[int, str],
    stop_boundary_text: str = '"',
    visualize: bool = False,
) -> tuple[str, str]:
    """Generate a string, allowing escaped quotes and backslashes."""
    raw, text_so_far = _generate_value(
        text_so_far,
        model,
        token_text_cache,
        lambda current: {
            token_id
            for token_id, text in token_text_cache.items()
            if _valid_string_addition(current, text)
        },
        lambda current: not _pending_escape(current),
        stop_boundary_text,
        "string",
        visualize,
    )
    return _unescape_string(raw), text_so_far


def generate_choice_value(
    text_so_far: str,
    model: Any,
    token_text_cache: dict[int, str],
    choices: list[str],
    stop_boundary_text: str,
    visualize: bool = False,
) -> tuple[str, str]:
    """Choose one value from a fixed list."""
    return _generate_value(
        text_so_far,
        model,
        token_text_cache,
        lambda current: get_valid_next_tokens(
            current, choices, token_text_cache
        ),
        lambda current: current in choices,
        stop_boundary_text,
        "choice",
        visualize,
    )


def generate_function_call(
    prompt: Prompt,
    functions: list[FunctionDefinition],
    model: Any,
    token_text_cache: dict[int, str],
    visualize: bool = False,
) -> FunctionCallResult:
    """Generate one schema-valid function call from a prompt."""
    text_so_far = build_prompt(prompt.prompt, functions) + '\n{"name": "'
    if visualize:
        print("Generating function name...")

    name, text_so_far = generate_choice_value(
        text_so_far,
        model,
        token_text_cache,
        [function.name for function in functions],
        '"',
        visualize,
    )
    text_so_far += '"'

    function = next((item for item in functions if item.name == name), None)
    if function is None:
        raise ValueError(f"Unknown function name: {name!r}")

    text_so_far += ', "parameters": {'
    parameters: dict[str, bool | str | float | int] = {}
    items = list(function.parameters.items())

    for index, (param_name, schema) in enumerate(items):
        separator = "}" if index == len(items) - 1 else ","
        text_so_far += f'"{param_name}": '
        if visualize:
            print(f"Generating {param_name} ({schema.type})...")

        if schema.type == "string":
            text_so_far += '"'
            value, text_so_far = generate_string_value(
                text_so_far, model, token_text_cache, visualize=visualize
            )
            text_so_far += '"'
        elif schema.type in ("number", "integer"):
            value, text_so_far = generate_numeric_value(
                text_so_far,
                model,
                token_text_cache,
                separator,
                schema.type == "integer",
                visualize,
            )
        elif schema.type == "boolean":
            text_value, text_so_far = generate_choice_value(
                text_so_far,
                model,
                token_text_cache,
                ["true", "false"],
                separator,
                visualize,
            )
            value = text_value == "true"
        else:
            raise ValueError(f"Unsupported parameter type: {schema.type!r}")

        parameters[param_name] = value
        text_so_far += separator
        if visualize:
            print(f"  {param_name}: {value!r}")

    if not items:
        text_so_far += "}"
    return FunctionCallResult(
        prompt=prompt.prompt,
        name=name,
        parameters=parameters,
    )
