# ABOUTME: The full generation loop: builds the fixed JSON template around a function's
# ABOUTME: parameters and fills each blank via constrained decoding into one function call.

from typing import Any, Protocol

from pydantic import BaseModel

from src.constrained_decoding import compute_valid_token_ids, mask_logits, select_next_token
from src.decoding_states import ChoiceState, DecodingState, NumberState, StringState
from src.models import FunctionDefinition

MAX_BLANK_TOKENS = 64  # safety cap: a blank that never completes must not hang forever


class GenerationError(Exception):
    """Raised when a prompt cannot be translated into a valid function call."""


class GenerationModel(Protocol):
    """Methods generation.py depends on from llm_sdk.Small_LLM_Model."""

    def encode(self, text: str) -> Any: ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: ...


class FunctionCallResult(BaseModel):
    """The structured result of translating one prompt into a function call.

    Attributes:
        prompt: The original natural-language prompt.
        name: The chosen function's name.
        parameters: The generated argument values, keyed by parameter name.
    """

    prompt: str
    name: str
    parameters: dict[str, float | str | bool]


def _build_instruction_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the natural-language context shown to the model before generation."""
    lines = [
        "You are a function-calling assistant. Given a user request, choose the "
        "single best function and its arguments from the list below.",
        "",
        "Available functions:",
    ]
    for fn in functions:
        params = ", ".join(f"{p}: {s.type}" for p, s in fn.parameters.items())
        lines.append(f"- {fn.name}({params}): {fn.description}")
    lines.append("")
    lines.append(f'User request: "{prompt}"')
    return "\n".join(lines)


def _state_for_type(json_type: str) -> DecodingState:
    """Return the DecodingState matching a parameter's declared JSON type."""
    if json_type == "number":
        return NumberState()
    if json_type == "string":
        return StringState()
    if json_type == "boolean":
        return ChoiceState(["true", "false"])
    raise GenerationError(f"Unsupported parameter type: {json_type}")


def _coerce_value(json_type: str, text: str) -> float | str | bool:
    """Convert a blank's accumulated text into the Python value it represents."""
    if json_type == "number":
        return float(text)
    if json_type == "boolean":
        return text == "true"
    return text


def _fill_blank(
    state: DecodingState,
    model: GenerationModel,
    token_text_cache: dict[int, str],
    text_so_far: str,
    stop_boundary_text: str,
) -> tuple[str, str]:
    """Generate tokens for one blank, letting the model choose when to stop.

    At each step, the candidate set is the valid continuation tokens for
    `state`, plus — only once the blank is already allowed to end — the
    token(s) matching the fixed template text that comes right after this
    blank. If the model's highest-scoring choice is that boundary text, the
    blank ends here (the boundary itself is injected by the caller, not
    consumed as part of the value).

    Args:
        state: The DecodingState governing what's valid for this blank.
        model: The LLM wrapper (encode + get_logits_from_input_ids).
        token_text_cache: Precomputed {token_id: decoded_text}.
        text_so_far: Full text generated so far (prompt + template + blanks).
        stop_boundary_text: The literal text that immediately follows this
            blank in the template (e.g. '"', ',', or '}').

    Returns:
        (the blank's final text, the updated text_so_far not including the
        boundary text itself).

    Raises:
        GenerationError: If no valid token exists, or the blank never
            completes within MAX_BLANK_TOKENS steps.
    """
    for _ in range(MAX_BLANK_TOKENS):
        input_ids = model.encode(text_so_far).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)

        valid_ids = compute_valid_token_ids(state, token_text_cache)
        boundary_ids: set[int] = set()
        if state.can_stop():
            boundary_ids = {
                token_id
                for token_id, text in token_text_cache.items()
                if text == stop_boundary_text
            }

        candidate_ids = valid_ids | boundary_ids
        if not candidate_ids:
            raise GenerationError("No valid continuation for the current blank")

        masked = mask_logits(logits, candidate_ids)
        next_id = select_next_token(masked)

        if next_id in boundary_ids and next_id not in valid_ids:
            return state.generated_text, text_so_far

        addition = token_text_cache[next_id]
        state.accept(addition)
        text_so_far += addition

    raise GenerationError("Blank did not complete within the token limit")


def generate_function_call(
    prompt: str,
    model: GenerationModel,
    token_text_cache: dict[int, str],
    functions: list[FunctionDefinition],
) -> FunctionCallResult:
    """Translate one natural-language prompt into a validated function call.

    Args:
        prompt: The user's natural-language request.
        model: The LLM wrapper used for encode()/get_logits_from_input_ids().
        token_text_cache: Precomputed {token_id: decoded_text}.
        functions: The available function definitions to choose from.

    Returns:
        The resulting FunctionCallResult.

    Raises:
        GenerationError: If the model selects an unknown function name or
            any blank fails to generate.
    """
    text_so_far = _build_instruction_prompt(prompt, functions)

    # --- Blank 1: the function name, injected into a fixed template ---
    text_so_far += '\n{"name": "'
    name_state = ChoiceState([fn.name for fn in functions])
    name_text, text_so_far = _fill_blank(
        name_state, model, token_text_cache, text_so_far, stop_boundary_text='"'
    )
    text_so_far += '"'

    function_def = next((fn for fn in functions if fn.name == name_text), None)
    if function_def is None:
        raise GenerationError(f"Model produced an unknown function name: {name_text!r}")

    # --- Blanks: one per parameter, in schema order ---
    text_so_far += ', "parameters": {'
    parameters: dict[str, float | str | bool] = {}
    param_items = list(function_def.parameters.items())

    for index, (param_name, schema) in enumerate(param_items):
        is_last = index == len(param_items) - 1
        separator = "}" if is_last else ","
        is_string = schema.type == "string"

        text_so_far += f'"{param_name}": '
        if is_string:
            text_so_far += '"'

        value_state = _state_for_type(schema.type)
        boundary = '"' if is_string else separator
        value_text, text_so_far = _fill_blank(
            value_state, model, token_text_cache, text_so_far, stop_boundary_text=boundary
        )

        if is_string:
            text_so_far += '"'
        parameters[param_name] = _coerce_value(schema.type, value_text)
        text_so_far += separator

    if not param_items:
        text_so_far += "}"  # close an empty parameters object
    text_so_far += "}"  # close the outer object

    return FunctionCallResult(prompt=prompt, name=name_text, parameters=parameters)