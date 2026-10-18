# ABOUTME: The full generation loop: builds the fixed JSON template around a function's
# ABOUTME: parameters and fills each blank via constrained decoding into one function call.
#
# WHY THIS FILE EXISTS (read this first):
# We want to build exactly this kind of text:
#   {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
# Everything EXCEPT the values in quotes/numbers is FIXED text that WE type
# ourselves (not generated). Only the blanks (name, a, b, ...) are filled by
# asking the model, one constrained token at a time, using the machinery
# from decoding_states.py and constrained_decoding.py.

from typing import Any, Protocol

from pydantic import BaseModel

from src.constrained_decoding import compute_valid_token_ids, mask_logits, select_next_token
from src.decoding_states import (
    ChoiceState,
    DecodingState,
    IntegerState,
    NumberState,
    StringState,
)
from src.models import FunctionDefinition
from src.visualization import GenerationStep, GenerationTracer

# Safety cap: if a blank somehow never completes (a bug, or a truly stuck
# model), stop after this many tokens instead of looping forever.
MAX_BLANK_TOKENS = 64


class GenerationError(Exception):
    """Raised when a prompt cannot be translated into a valid function call."""


class GenerationModel(Protocol):
    """The only two model abilities this file needs (see constrained_decoding.py
    for why we use a Protocol instead of importing the real model class).
    """

    def encode(self, text: str) -> Any: ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: ...


class FunctionCallResult(BaseModel):
    """The final, structured answer for one prompt - this is exactly what
    gets written (as JSON) into the output file.
    """

    prompt: str
    name: str
    parameters: dict[str, float | int | str | bool]


def _build_instruction_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the plain-English text shown to the model before we start
    constraining anything. This just gives the model context (what
    functions exist, what the user asked) - it is NOT part of the masking
    mechanism itself.
    """
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
    """Pick the right "rulebook" (DecodingState) for a parameter's declared
    JSON type. This is the ONLY place that needs to change if we ever add
    support for a new type.
    """
    if json_type == "number":
        return NumberState()
    if json_type == "integer":
        return IntegerState()
    if json_type == "string":
        return StringState()
    if json_type == "boolean":
        return ChoiceState(["true", "false"])
    raise GenerationError(f"Unsupported parameter type: {json_type}")


def _extract_value(json_type: str, state: DecodingState) -> float | int | str | bool:
    """Once a blank is finished, turn its raw text into the real Python
    value we'll put in the output (a float, an int, a string, or a bool).

    Args:
        json_type: The schema's declared type for this blank.
        state: The (now finished) DecodingState that generated it.

    Returns:
        The typed Python value.

    Raises:
        GenerationError: If json_type isn't one we know how to handle.
    """
    if json_type == "number":
        return float(state.generated_text)
    if json_type == "integer":
        return int(state.generated_text)
    if json_type == "boolean":
        return state.generated_text == "true"
    if json_type == "string":
        # Strings are special: generated_text may still contain raw escape
        # sequences (\" or \\) - see decoding_states.StringState for why.
        # unescape() turns that into the real, final string value.
        if not isinstance(state, StringState):
            raise GenerationError("Internal error: string type with wrong state")
        return state.unescape()
    raise GenerationError(f"Unsupported parameter type: {json_type}")


def _fill_blank(
    state: DecodingState,
    model: GenerationModel,
    token_text_cache: dict[int, str],
    text_so_far: str,
    stop_boundary_text: str,
    blank_label: str = "value",
    tracer: GenerationTracer | None = None,
) -> tuple[str, str]:
    """Generate tokens for ONE blank, one step at a time, until it's done.

    --------------------------------------------------------------------
    THE TRICKY PART - please read this before touching this function:

    How does the model know WHEN to stop? Example: for a number, after
    typing "2" we already have a valid number. But maybe the right answer
    is "265" and we should keep going! We don't want to hardcode a rule
    like "always stop after 1 digit" - that would often be wrong.

    So instead, we let the model itself choose, by playing fair: at every
    step we offer TWO kinds of options together, and let argmax (the normal
    "pick the highest score" step) decide between them:
        - "keep going" options: any token that's still a valid continuation
          of the value (e.g. another digit).
        - ONE "stop here" option: the token matching whatever fixed text
          comes right after this blank in the template (e.g. the comma
          "," or the closing quote '"').
    Whichever one the model actually scores highest, wins - exactly like
    any other constrained-decoding step. If "stop" wins, we end the blank
    here (the boundary text itself gets added by the CALLER afterwards, not
    by this function - we only use it as a comparison point).
    --------------------------------------------------------------------

    Args:
        state: The DecodingState (rulebook) for this blank.
        model: The LLM wrapper (encode + get_logits_from_input_ids).
        token_text_cache: Precomputed {token_id: decoded_text}.
        text_so_far: ALL text generated so far (prompt + fixed template +
            previous blanks). We re-encode this whole string at every step
            (see note below) instead of re-using old token IDs.
        stop_boundary_text: The literal text that comes right after this
            blank in the template (e.g. '"', ',', or '}').
        blank_label: Just a human-readable name for this blank, used only
            for the optional visualization/trace output.
        tracer: Optional GenerationTracer; pass None to generate silently.

    Returns:
        (the blank's final raw text, the updated text_so_far - NOT
        including the boundary text, which the caller adds itself).

    Raises:
        GenerationError: If no valid token exists at some step, or the
            blank never finishes within MAX_BLANK_TOKENS steps.
    """
    for _ in range(MAX_BLANK_TOKENS):
        # NOTE on re-encoding the full text every step (instead of just
        # appending new token IDs to a growing list): tokenizers don't
        # always split text the same way when it's glued together from
        # separate pieces versus encoded all at once (this is a known BPE
        # quirk). Re-encoding the whole string each time sidesteps that
        # problem entirely. It costs one extra (cheap) encode() call per
        # step - tiny compared to the model's forward pass itself.
        input_ids = model.encode(text_so_far).tolist()[0]
        logits = model.get_logits_from_input_ids(input_ids)

        # Step 1: which tokens are valid "keep going" choices?
        valid_ids = compute_valid_token_ids(state, token_text_cache)

        # Step 2: are we ALSO allowed to offer "stop here" as a choice?
        # Only once the value is already complete (can_stop() == True) -
        # otherwise we'd risk stopping on an incomplete/invalid value.
        boundary_ids: set[int] = set()
        if state.can_stop():
            boundary_ids = {
                token_id
                for token_id, text in token_text_cache.items()
                if text == stop_boundary_text
            }

        # Step 3: mask + argmax over BOTH sets of options together.
        candidate_ids = valid_ids | boundary_ids
        if not candidate_ids:
            raise GenerationError("No valid continuation for the current blank")
        masked = mask_logits(logits, candidate_ids)
        next_id = select_next_token(masked)

        chosen_text = token_text_cache[next_id]
        # The model "chose to stop" only if it picked a boundary token that
        # is NOT also a valid content token (keeps this unambiguous).
        stopped = next_id in boundary_ids and next_id not in valid_ids

        if tracer is not None:
            tracer.record(
                GenerationStep(
                    blank_label=blank_label,
                    text_tail=text_so_far[-20:],
                    num_value_candidates=len(valid_ids),
                    num_stop_candidates=len(boundary_ids),
                    chosen_text=chosen_text,
                    stopped=stopped,
                )
            )

        if stopped:
            return state.generated_text, text_so_far

        # Otherwise: commit this token as part of the value, and loop again.
        state.accept(chosen_text)
        text_so_far += chosen_text

    raise GenerationError("Blank did not complete within the token limit")


def generate_function_call(
    prompt: str,
    model: GenerationModel,
    token_text_cache: dict[int, str],
    functions: list[FunctionDefinition],
    tracer: GenerationTracer | None = None,
) -> FunctionCallResult:
    """Translate ONE natural-language prompt into ONE validated function call.

    This is the "assembly line": it writes the fixed template pieces
    directly (no model involved), and calls _fill_blank() for every actual
    blank, in order: first the function name, then each parameter value.

    Args:
        prompt: The user's natural-language request.
        model: The LLM wrapper.
        token_text_cache: Precomputed {token_id: decoded_text}.
        functions: The available function definitions to choose from.
        tracer: Optional GenerationTracer for the visualization bonus.

    Returns:
        The resulting FunctionCallResult.

    Raises:
        GenerationError: If the model picks an unknown function name, or
            any blank fails to generate.
    """
    text_so_far = _build_instruction_prompt(prompt, functions)

    # ---- Blank 1: the function name ----
    text_so_far += '\n{"name": "'  # fixed text: WE write this, not the model
    name_state = ChoiceState([fn.name for fn in functions])
    name_text, text_so_far = _fill_blank(
        name_state,
        model,
        token_text_cache,
        text_so_far,
        stop_boundary_text='"',  # a closing quote comes right after the name
        blank_label="name",
        tracer=tracer,
    )
    text_so_far += '"'  # now we add that closing quote ourselves

    # Look up the actual FunctionDefinition so we know which parameters
    # (and types) to generate next.
    function_def = next((fn for fn in functions if fn.name == name_text), None)
    if function_def is None:
        # Should not normally happen (ChoiceState only allows known names),
        # but we check anyway instead of trusting it blindly.
        raise GenerationError(f"Model produced an unknown function name: {name_text!r}")

    # ---- Blanks 2..N: one per parameter, in the schema's order ----
    text_so_far += ', "parameters": {'
    parameters: dict[str, float | int | str | bool] = {}
    param_items = list(function_def.parameters.items())

    for index, (param_name, schema) in enumerate(param_items):
        is_last = index == len(param_items) - 1
        # The character that closes this value: "," for a middle parameter,
        # "}" for the last one (closing the whole "parameters" object).
        separator = "}" if is_last else ","
        is_string = schema.type == "string"

        text_so_far += f'"{param_name}": '  # fixed text: the parameter's key
        if is_string:
            text_so_far += '"'  # opening quote, fixed text (strings only)

        value_state = _state_for_type(schema.type)
        # For a string, the very next fixed character is the closing quote.
        # For everything else, it's directly the separator (comma/brace).
        boundary = '"' if is_string else separator
        _, text_so_far = _fill_blank(
            value_state,
            model,
            token_text_cache,
            text_so_far,
            stop_boundary_text=boundary,
            blank_label=param_name,
            tracer=tracer,
        )

        if is_string:
            text_so_far += '"'  # closing quote, fixed text
        parameters[param_name] = _extract_value(schema.type, value_state)
        text_so_far += separator  # fixed text: "," or the closing "}"

    if not param_items:
        # A function with zero parameters: the loop above never ran, so we
        # still need to close the (empty) "parameters": {} ourselves.
        text_so_far += "}"
    text_so_far += "}"  # close the OUTER {"name": ..., "parameters": ...} object

    if tracer is not None:
        tracer.print_summary()

    return FunctionCallResult(prompt=prompt, name=name_text, parameters=parameters)