# ABOUTME: Generates function calls from prompts using constrained decoding.
# ABOUTME: Builds a fixed JSON template and fills each blank (name/number/integer/
# ABOUTME: string/boolean) one token at a time, letting the model itself choose
# ABOUTME: when each value is complete by competing a "stop" token against continuation.

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
    """Check whether `text` could still become a valid number/integer."""
    if text in ("", "-"):
        return True
    seen_dot = False
    for index, char in enumerate(text):
        if char == "-":
            if index != 0:
                return False
        elif char == "." and allow_dot:
            if seen_dot:
                return False
            seen_dot = True
        elif not char.isdigit():
            return False
    return True


def _is_complete_number(text: str) -> bool:
    """Check whether `text` is already a complete, valid number/integer."""
    if text in ("", "-") or text.endswith("."):
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


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
        if _is_complete_number(generated):
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
            return value, text_so_far

        addition = token_text_cache[next_id]
        generated += addition
        text_so_far += addition

    raise ValueError(
        f"Numeric value did not complete within the token limit (partial: {generated!r})"
    )


# --- String values (with JSON escape support) -------------------------------

def _pending_escape(text: str) -> bool:
    """True if `text` ends with an odd run of backslashes (escape open)."""
    count = 0
    for char in reversed(text):
        if char != "\\":
            break
        count += 1
    return count % 2 == 1


def _is_valid_string_addition(generated: str, addition: str) -> bool:
    """Check whether `addition` can be appended to `generated` string content.

    A literal '"' or '\\' inside the value must arrive as the standard JSON
    escape (\\" or \\\\); a *bare* unescaped '"' always ends the string.
    """
    text = generated
    for char in addition:
        if _pending_escape(text):
            if char not in ('"', "\\"):
                return False
        elif char == '"':
            return False
        text += char
    return True


def _find_repeating_tail(
    text: str, min_period: int = 1, max_period: int = 20, min_repeats: int = 3
) -> tuple[int, int] | None:
    """Detect a short pattern repeating at the very end of `text`.

    Small models under pure greedy decoding can fall into a degenerate loop
    (e.g. "cat.*cat.*cat.*..." forever) where the "continue" token keeps
    outscoring the "stop" token even though the value was already complete.
    This never happens in NumberState/ChoiceState (both are bounded by
    construction), but an open-ended string value has no such bound.

    Args:
        text: The text to inspect (the string value generated so far).
        min_period: Shortest repeating unit length to consider.
        max_period: Longest repeating unit length to consider.
        min_repeats: Minimum number of consecutive repeats to count as a loop.

    Returns:
        (period, repeat_count) for the longest qualifying repetition found at
        the tail of `text`, or None if nothing qualifies.
    """
    n = len(text)
    best: tuple[int, int] | None = None
    for period in range(min_period, max_period + 1):
        if period * min_repeats > n:
            break
        unit = text[n - period:]
        repeats = 1
        pos = n - period
        while pos - period >= 0 and text[pos - period:pos] == unit:
            repeats += 1
            pos -= period
        if repeats >= min_repeats and (best is None or repeats * period > best[0] * best[1]):
            best = (period, repeats)
    return best


def _unescape_string(text: str) -> str:
    """Decode the raw generated text (with \\" and \\\\ escapes) into the
    actual string value.
    """
    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] in ('"', "\\"):
            result.append(text[i + 1])
            i += 2
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


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

        # A small model under pure greedy decoding can fall into a repeating
        # loop (e.g. "cat.*cat.*cat.*...") where "continue" keeps outscoring
        # "stop" forever, even once stopping is a legal option. Detect that
        # and force a stop, trimming back to a single clean copy of the
        # repeated unit, rather than waiting out the full token budget (and
        # skipping the model call for this step, since we already know what
        # we're about to do).
        if boundary_ids:
            repetition = _find_repeating_tail(generated)
            if repetition is not None:
                period, repeats = repetition
                removed = period * (repeats - 1)
                generated = generated[: len(generated) - removed]
                text_so_far = text_so_far[: len(text_so_far) - removed]
                if visualize:
                    print(
                        f"    [string] detected repeating pattern "
                        f"(period={period}, x{repeats}) - stopping early with "
                        f"{generated!r}"
                    )
                return _unescape_string(generated), text_so_far

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