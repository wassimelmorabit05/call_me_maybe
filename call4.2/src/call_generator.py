# ABOUTME: Generates function calls from prompts using constrained decoding.
# ABOUTME: Builds a fixed JSON template and fills each blank (name/number/integer/
# ABOUTME: string/boolean) one token at a time, letting the model itself choose
# ABOUTME: when each value is complete by competing a "stop" token against continuation.

import json
import math
import re
from typing import Any

from .decoding import get_valid_next_tokens, mask_logits, select_next_token
from .input_loader import FunctionDefinition, Prompt
from .models import FunctionCallResult

MAX_VALUE_TOKENS = 64  # safety cap: a value that never completes must not hang forever


def build_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the natural-language context shown to the model before generation.

    Args:
        user_prompt: The user's natural-language request.
        functions: The available function definitions to choose from.

    Returns:
        The instruction text, ending right before the JSON template begins.
    """
    lines = [
        "You are a function calling assistant. Given a user request, choose "
        "the single best function and its arguments from the list below.",
        "",
        "Available functions:",
    ]
    for func in functions:
        params = ", ".join(f"{name} ({p.type})" for name, p in func.parameters.items())
        lines.append(f"- {func.name}({params}): {func.description}")
    lines.append("")
    lines.append(f'User request: "{user_prompt}"')
    return "\n".join(lines)


# --- Number / integer values ------------------------------------------------

def _is_number_prefix(text: str, allow_dot: bool) -> bool:
    """Return whether text is a prefix of a valid JSON number.

    For the integer schema type, decimal and exponent notation are disabled.
    For number, JSON decimal and exponent notation are accepted.
    """
    if text in ("", "-"):
        return True

    index = 1 if text.startswith("-") else 0
    if index == len(text):
        return True

    # JSON integer part: either exactly zero, or a non-zero digit followed
    # by any number of ASCII digits. This rejects leading-zero forms such as 01.
    if text[index] == "0":
        index += 1
        if index < len(text) and text[index].isdigit() and text[index].isascii():
            return False
    elif "1" <= text[index] <= "9":
        index += 1
        while index < len(text) and "0" <= text[index] <= "9":
            index += 1
    else:
        return False

    fraction_digits = 0
    if allow_dot and index < len(text) and text[index] == ".":
        index += 1
        while index < len(text) and "0" <= text[index] <= "9":
            index += 1
            fraction_digits += 1

    if allow_dot and index < len(text) and text[index] in "eE":
        # Once a decimal point has appeared, at least one fractional digit
        # must precede an exponent ("1.e2" is not a valid JSON number).
        if "." in text[:index] and fraction_digits == 0:
            return False
        index += 1
        if index < len(text) and text[index] in "+-":
            index += 1
        while index < len(text) and "0" <= text[index] <= "9":
            index += 1

    return index == len(text)


def _is_complete_number(text: str, allow_dot: bool = True) -> bool:
    """Check whether text is a complete JSON number (or JSON integer)."""
    if allow_dot:
        pattern = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
    else:
        pattern = r"-?(?:0|[1-9][0-9]*)"
    return re.fullmatch(pattern, text) is not None


def generate_numeric_value(
    text_so_far: str,
    model: Any,
    token_text_cache: dict[int, str],
    stop_boundary_text: str,
    is_integer: bool,
    visualize: bool = False,
) -> tuple[float | int, str]:
    """Generate a number or integer value, letting the model decide when to stop.

    At each step, valid digit/sign/dot continuations compete against the
    literal template text that follows the value (e.g. "," or "}") - once
    the accumulated digits already form a complete number, whichever the
    model scores higher wins. This handles negative numbers, multi-digit
    numbers, and decimals uniformly, with no special-casing or text-scraping
    from the original prompt.

    Args:
        text_so_far: Full text generated so far (prompt + template + blanks).
        model: The LLM wrapper (encode + get_logits_from_input_ids).
        token_text_cache: Precomputed {token_id: decoded_text}.
        stop_boundary_text: The literal text immediately following this value
            in the template (e.g. "," or "}").
        is_integer: If True, a decimal point is never allowed (schema type
            "integer"); otherwise it is (schema type "number").

    Returns:
        (the generated value as int or float, the updated text_so_far).

    Raises:
        ValueError: If no valid continuation exists, or the value never
            completes within MAX_VALUE_TOKENS steps.
    """
    allow_dot = not is_integer
    generated = ""

    for _ in range(MAX_VALUE_TOKENS):
        input_ids = model.encode(text_so_far).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)

        valid_ids = {
            token_id
            for token_id, text in token_text_cache.items()
            if _is_number_prefix(generated + text, allow_dot)
        }
        boundary_ids: set[int] = set()
        if _is_complete_number(generated, allow_dot=allow_dot):
            boundary_ids = {
                token_id
                for token_id, text in token_text_cache.items()
                if text == stop_boundary_text
            }

        candidate_ids = valid_ids | boundary_ids
        if not candidate_ids:
            raise ValueError("No valid continuation while generating a number")

        masked = mask_logits(logits, candidate_ids)
        next_id = select_next_token(masked)
        stopped = next_id in boundary_ids and next_id not in valid_ids

        if visualize:
            print(
                f"    [number] candidates={len(candidate_ids)} "
                f"(value={len(valid_ids)}, stop={len(boundary_ids)})  "
                f"{'STOP' if stopped else 'PICK'} -> {token_text_cache[next_id]!r}"
            )

        if stopped:
            value = int(generated) if is_integer else float(generated)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Number is outside the finite float range: {generated!r}")
            return value, text_so_far

        addition = token_text_cache[next_id]
        generated += addition
        text_so_far += addition

    raise ValueError(
        f"Numeric value did not complete within the token limit (partial: {generated!r})"
    )


# --- String values (with JSON escape support) -------------------------------

def _pending_escape(text: str) -> bool:
    """Return True when text ends inside an incomplete JSON escape."""
    index = 0
    while index < len(text):
        if text[index] != "\\":
            if ord(text[index]) < 0x20:
                return True
            index += 1
            continue
        index += 1
        if index >= len(text):
            return True
        escape = text[index]
        index += 1
        if escape == "u":
            for _ in range(4):
                if index >= len(text):
                    return True
                if text[index] not in "0123456789abcdefABCDEF":
                    return True
                index += 1
        elif escape not in '"\\/bfnrt':
            return True
    return False


def _is_valid_string_addition(generated: str, addition: str) -> bool:
    """Check that generated + addition is a valid prefix of JSON string content.

    Supports all JSON escapes and rejects unescaped control characters.
    An incomplete escape is allowed as a prefix because the next token may
    complete it.
    """
    text = generated + addition
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            return False
        if ord(char) < 0x20:
            return False
        if char != "\\":
            index += 1
            continue

        index += 1
        if index >= len(text):
            return True
        escape = text[index]
        index += 1
        if escape == "u":
            for _ in range(4):
                if index >= len(text):
                    return True
                if text[index] not in "0123456789abcdefABCDEF":
                    return False
                index += 1
        elif escape not in '"\\/bfnrt':
            return False
    return True




def _unescape_string(text: str) -> str:
    """Decode JSON string content, including standard escapes and Unicode."""
    return json.loads('"' + text + '"')


def generate_string_value(
    text_so_far: str,
    model: Any,
    token_text_cache: dict[int, str],
    stop_boundary_text: str = '"',
    visualize: bool = False,
) -> tuple[str, str]:
    """Generate a string value, letting the model decide when to stop.

    Supports literal quotes and backslashes in the value (e.g. a regex like
    "\\d+", or a Windows path like "C:\\Users\\john") via standard JSON
    escaping, while a bare unescaped '"' reliably ends the string.

    Args:
        text_so_far: Full text generated so far (prompt + template + blanks).
        model: The LLM wrapper (encode + get_logits_from_input_ids).
        token_text_cache: Precomputed {token_id: decoded_text}.
        stop_boundary_text: The literal text that ends the string (the
            closing quote, '"').

    Returns:
        (the decoded string value, the updated text_so_far).

    Raises:
        ValueError: If no valid continuation exists, or the value never
            completes within MAX_VALUE_TOKENS steps.
    """
    generated = ""

    for _ in range(MAX_VALUE_TOKENS):
        pending_escape = _pending_escape(generated)
        boundary_ids: set[int] = set()
        if not pending_escape:
            boundary_ids = {
                token_id
                for token_id, text in token_text_cache.items()
                if text == stop_boundary_text
            }

        input_ids = model.encode(text_so_far).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)

        valid_ids = {
            token_id
            for token_id, text in token_text_cache.items()
            if _is_valid_string_addition(generated, text)
        }

        candidate_ids = valid_ids | boundary_ids
        if not candidate_ids:
            raise ValueError("No valid continuation while generating a string")

        masked = mask_logits(logits, candidate_ids)
        next_id = select_next_token(masked)
        stopped = next_id in boundary_ids and next_id not in valid_ids

        if visualize:
            print(
                f"    [string] candidates={len(candidate_ids)} "
                f"(value={len(valid_ids)}, stop={len(boundary_ids)})  "
                f"{'STOP' if stopped else 'PICK'} -> {token_text_cache[next_id]!r}  "
                f"so_far={generated!r}"
            )

        if stopped:
            return _unescape_string(generated), text_so_far

        addition = token_text_cache[next_id]
        generated += addition
        text_so_far += addition

    raise ValueError(
        f"String value did not complete within the token limit (partial: {generated!r})"
    )


# --- Choice values (function name, booleans) --------------------------------

def generate_choice_value(
    text_so_far: str,
    model: Any,
    token_text_cache: dict[int, str],
    choices: list[str],
    stop_boundary_text: str,
    visualize: bool = False,
) -> tuple[str, str]:
    """Generate a value constrained to a fixed, known set of strings.

    Used for the function name (choices = known function names) and for
    boolean values (choices = ["true", "false"]).

    Args:
        text_so_far: Full text generated so far (prompt + template + blanks).
        model: The LLM wrapper (encode + get_logits_from_input_ids).
        token_text_cache: Precomputed {token_id: decoded_text}.
        choices: The allowed complete values.
        stop_boundary_text: The literal text that follows this value in the
            template once it is complete.

    Returns:
        (the chosen value, the updated text_so_far).

    Raises:
        ValueError: If no valid continuation exists, or no choice completes
            within MAX_VALUE_TOKENS steps.
    """
    generated = ""

    for _ in range(MAX_VALUE_TOKENS):
        input_ids = model.encode(text_so_far).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)

        valid_ids = get_valid_next_tokens(generated, choices, token_text_cache)
        boundary_ids: set[int] = set()
        if generated in choices:
            boundary_ids = {
                token_id
                for token_id, text in token_text_cache.items()
                if text == stop_boundary_text
            }

        candidate_ids = valid_ids | boundary_ids
        if not candidate_ids:
            raise ValueError(f"No valid continuation among choices {choices}")

        masked = mask_logits(logits, candidate_ids)
        next_id = select_next_token(masked)
        stopped = next_id in boundary_ids and next_id not in valid_ids

        if visualize:
            print(
                f"    [choice] candidates={len(candidate_ids)} "
                f"(value={len(valid_ids)}, stop={len(boundary_ids)})  "
                f"{'STOP' if stopped else 'PICK'} -> {token_text_cache[next_id]!r}  "
                f"so_far={generated!r}"
            )

        if stopped:
            return generated, text_so_far

        addition = token_text_cache[next_id]
        generated += addition
        text_so_far += addition

    raise ValueError(
        f"Choice did not complete within the token limit (partial: {generated!r})"
    )


# --- Full pipeline -----------------------------------------------------------

def generate_function_call(
    prompt: Prompt,
    functions: list[FunctionDefinition],
    model: Any,
    token_text_cache: dict[int, str],
    visualize: bool = False,
) -> FunctionCallResult:
    """Translate one natural-language prompt into a validated function call.

    Args:
        prompt: The validated user prompt.
        functions: The available function definitions to choose from.
        model: The LLM wrapper used for encode()/get_logits_from_input_ids().
        token_text_cache: Precomputed {token_id: decoded_text}.
        visualize: If True, print a short trace of each generation step.

    Returns:
        The resulting FunctionCallResult.

    Raises:
        ValueError: If the model selects an unknown function name, or any
            value fails to generate (no valid continuation, or runaway
            generation past the token limit).
    """
    green, yellow, reset = "\033[32m", "\033[33m", "\033[0m"

    text_so_far = build_prompt(prompt.prompt, functions)

    # --- the function name, injected into a fixed template ---
    text_so_far += '\n{"name": "'
    if visualize:
        print(f"  {yellow}Generating function name...{reset}")
    name, text_so_far = generate_choice_value(
        text_so_far,
        model,
        token_text_cache,
        [fn.name for fn in functions],
        '"',
        visualize=visualize,
    )
    text_so_far += '"'
    if visualize:
        print(f"    {green}-> {name}{reset}")

    function_def = next((fn for fn in functions if fn.name == name), None)
    if function_def is None:
        raise ValueError(f"Model produced an unknown function name: {name!r}")

    # --- one blank per parameter, in schema order ---
    text_so_far += ', "parameters": {'
    parameters: dict[str, bool | str | float | int] = {}
    items = list(function_def.parameters.items())

    for index, (param_name, schema) in enumerate(items):
        is_last = index == len(items) - 1
        separator = "}" if is_last else ","
        text_so_far += f'"{param_name}": '

        if visualize:
            print(f"  {yellow}Generating {param_name} ({schema.type})...{reset}")

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
                stop_boundary_text=separator,
                is_integer=(schema.type == "integer"),
                visualize=visualize,
            )
        elif schema.type == "boolean":
            text_value, text_so_far = generate_choice_value(
                text_so_far,
                model,
                token_text_cache,
                ["true", "false"],
                separator,
                visualize=visualize,
            )
            value = text_value == "true"
        else:
            raise ValueError(f"Unsupported parameter type: {schema.type!r}")

        if visualize:
            print(f"  {yellow}{param_name}:{reset} {green}{value!r}{reset}")

        parameters[param_name] = value
        text_so_far += separator

    if not items:
        text_so_far += "}"  # close an empty parameters object
    text_so_far += "}"  # close the outer object

    return FunctionCallResult(prompt=prompt.prompt, name=name, parameters=parameters)