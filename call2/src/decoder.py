import re

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition
from .prompts import build_arguments_prompt
from typing import cast


STRING_PREFIX = re.compile(
    r'(?:[^"\\]|\\[\\/bfnrt])*\\?'
)
STRING_COMPLETE = re.compile(
    r'(?:[^"\\]|\\[\\/bfnrt])*'
)


def select_best_valid_token(
    logits: list[float],
    valid_tokens: list[int],
) -> int:
    """Select the valid token with the highest logit."""

    if not valid_tokens:
        raise ValueError("No valid tokens available.")

    best_token = valid_tokens[0]

    for token_id in valid_tokens[1:]:
        if logits[token_id] > logits[best_token]:
            best_token = token_id

    return best_token


def get_valid_next_tokens(
    candidates: list[list[int]],
    generated_tokens: list[int],
) -> list[int]:
    """Return valid next tokens for candidate sequences."""

    valid_tokens = []

    for candidate in candidates:
        prefix_length = len(generated_tokens)

        if candidate[:prefix_length] != generated_tokens:
            continue

        if prefix_length < len(candidate):
            next_token = candidate[prefix_length]

            if next_token not in valid_tokens:
                valid_tokens.append(next_token)

    return valid_tokens


def get_parameter_schema(
    function: FunctionDefinition,
) -> list[tuple[str, str]]:
    """Return parameter names and types in order."""

    return [
        (name, definition.type)
        for name, definition in function.parameters.items()
    ]


def is_number_prefix(value: str) -> bool:
    """Check whether text can be a number prefix."""

    if value in ("", "-"):
        return True

    try:
        float(value)
        return True
    except ValueError:
        return False


def can_end_number(value: str) -> bool:
    """Check whether a number can end."""

    if value in ("", "-"):
        return False

    try:
        float(value)
        return True
    except ValueError:
        return False


def get_number_end_token(
    schema: list[tuple[str, str]],
    parameter_index: int,
) -> str:
    """Return the token following a number."""

    if parameter_index == len(schema) - 1:
        return "}"

    return ","


def is_valid_number_candidate(
    current_value: str,
    token_text: str,
    schema: list[tuple[str, str]],
    parameter_index: int,
) -> bool:
    """Check whether a token is valid for a number."""

    candidate = current_value + token_text

    if is_number_prefix(candidate):
        return True

    if not can_end_number(current_value):
        return False

    return token_text == get_number_end_token(
        schema,
        parameter_index,
    )


def get_valid_number_token_ids(
    current_value: str,
    schema: list[tuple[str, str]],
    parameter_index: int,
    vocabulary: dict[str, int],
) -> list[int]:
    """Return valid token IDs for a number."""

    valid_tokens = []

    for token_text, token_id in vocabulary.items():
        if is_valid_number_candidate(
            current_value,
            token_text,
            schema,
            parameter_index,
        ):
            valid_tokens.append(token_id)

    return valid_tokens


def generate_number_value(
    model: Small_LLM_Model,
    prompt: str,
    generated_text: str,
    schema: list[tuple[str, str]],
    parameter_index: int,
    vocabulary: dict[str, int],
) -> tuple[str, str]:
    """Generate a number using constrained decoding."""

    current_value = ""

    while True:
        full_text = (
            prompt
            + generated_text
            + current_value
        )

        input_ids = model.encode(
            full_text
        )[0].tolist()

        logits = model.get_logits_from_input_ids(
            input_ids
        )

        valid_tokens = get_valid_number_token_ids(
            current_value,
            schema,
            parameter_index,
            vocabulary,
        )

        next_token_id = select_best_valid_token(
            logits,
            valid_tokens,
        )

        next_token = model.decode(
            [next_token_id]
        )

        end_token = get_number_end_token(
            schema,
            parameter_index,
        )

        if next_token == end_token:
            return current_value, end_token

        current_value += next_token


def is_valid_string_token(token_text: str) -> bool:
    """Check whether a token can continue a string value."""

    if token_text == "" or "\ufffd" in token_text:
        return False

    before_quote = token_text.split('"')[0]

    return all(
        ord(char) >= 0x20
        for char in before_quote
    )


def get_valid_string_token_ids(
    model: Small_LLM_Model,
    vocabulary: dict[str, int],
) -> list[int]:
    """Return valid token IDs for a JSON string."""

    valid_tokens = []

    for token_id in vocabulary.values():
        decoded_token = model.decode([token_id])

        if decoded_token == '"':
            valid_tokens.append(token_id)
            continue

        if is_valid_string_token(decoded_token):
            valid_tokens.append(token_id)

    return valid_tokens


def generate_string_value(
    model: Small_LLM_Model,
    prompt: str,
    generated_text: str,
    valid_string_tokens: list[int],
) -> str:
    """Generate a JSON string value using constrained decoding."""

    current_value = ""

    for _ in range(60):
        full_text = (
            prompt
            + generated_text
            + '"'
            + current_value
        )

        input_ids = model.encode(
            full_text
        )[0].tolist()

        logits = model.get_logits_from_input_ids(
            input_ids
        )

        ranked_tokens = sorted(
            valid_string_tokens,
            key=lambda token_id: logits[token_id],
            reverse=True,
        )

        for token_id in ranked_tokens:
            token_text = cast(str, model.decode([token_id]))
            before_quote, quote, _ = token_text.partition('"')
            candidate = current_value + before_quote

            if (
                quote
                and STRING_COMPLETE.fullmatch(candidate)
            ):
                return candidate

            if (
                not quote
                and STRING_PREFIX.fullmatch(candidate)
            ):
                current_value = candidate
                break
        else:
            raise RuntimeError(
                "No valid string token available."
            )

    raise RuntimeError(
        "String generation did not terminate."
    )


def generate_arguments(
    model: Small_LLM_Model,
    user_prompt: str,
    function: FunctionDefinition,
    vocabulary: dict[str, int],
) -> str:
    """Generate function arguments using constrained decoding."""

    prompt = build_arguments_prompt(
        user_prompt,
        function,
    )

    schema = get_parameter_schema(function)

    valid_string_tokens = []

    if any(
        parameter_type == "string"
        for _, parameter_type in schema
    ):
        valid_string_tokens = get_valid_string_token_ids(
            model,
            vocabulary,
        )

    generated_text = "{"

    for parameter_index, (
        name,
        parameter_type,
    ) in enumerate(schema):
        generated_text += f'"{name}":'

        if parameter_type in ("number", "integer"):
            value, end_token = generate_number_value(
                model=model,
                prompt=prompt,
                generated_text=generated_text,
                schema=schema,
                parameter_index=parameter_index,
                vocabulary=vocabulary,
            )

            generated_text += value
            generated_text += end_token

        elif parameter_type == "string":
            value = generate_string_value(
                model=model,
                prompt=prompt,
                generated_text=generated_text,
                valid_string_tokens=valid_string_tokens,
            )

            generated_text += f'"{value}"'

            if parameter_index == len(schema) - 1:
                generated_text += "}"
            else:
                generated_text += ","

        else:
            raise ValueError(
                "Unsupported parameter type: "
                f"{parameter_type}"
            )

    return generated_text
