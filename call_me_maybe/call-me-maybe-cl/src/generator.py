# ABOUTME: Turns one natural-language prompt into a FunctionCallResult by
# ABOUTME: applying constrained decoding per field: function name, then each
# ABOUTME: parameter value, typed according to functions_definition.json.
#
# Design decision (documented further in README.md "Algorithm explanation"):
# we do NOT ask the LLM to emit the raw JSON syntax ("{", ",", ":", "}")
# token by token. Instead we constrain the LLM only where the schema
# genuinely restricts the answer space (which function name, which digits
# make up a number, which of "true"/"false" for a boolean), and assemble
# the final object in Python with json.dumps (via pydantic's model_dump +
# io_utils.write_results). That guarantees 100% syntactic validity for
# free, while the LLM still makes every *semantic* decision: which function
# to call and what each parameter's value is. No heuristic ever picks the
# function name -- only masked argmax over the model's own logits does.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import FunctionCallResult, FunctionDef, ParamValue
from .vocab import Vocabulary

if TYPE_CHECKING:  # pragma: no cover
    from llm_sdk import Small_LLM_Model

MAX_NAME_TOKENS = 24
MAX_NUMBER_TOKENS = 24
MAX_STRING_TOKENS = 64
MAX_BOOL_TOKENS = 8

_STOP_STRINGS = {"\n", "\n\n", "\r\n"}


def _argmax(logits: list[float]) -> int:
    best_i = 0
    best_v = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best_v:
            best_v = logits[i]
            best_i = i
    return best_i


def _masked_argmax(logits: list[float], allowed_ids: set[int]) -> int | None:
    best_i: int | None = None
    best_v = float("-inf")
    for i in allowed_ids:
        if 0 <= i < len(logits) and logits[i] > best_v:
            best_v = logits[i]
            best_i = i
    return best_i


def _looks_numeric(s: str) -> bool:
    s = s.strip()
    return bool(s) and all(c.isdigit() or c in ".-" for c in s)


def _select_from_candidates(
    model: "Small_LLM_Model",
    vocab: Vocabulary,
    input_ids: list[int],
    candidates: list[str],
    max_tokens: int,
) -> str:
    """Generate text token-by-token, masking the logits at every step so
    only tokens that keep the partial string a valid prefix of one of
    `candidates` survive. This is the constrained-decoding core used for
    both function-name selection and boolean values."""
    generated = ""
    ids = list(input_ids)
    for _ in range(max_tokens):
        if generated in candidates:
            break
        logits = model.get_logits_from_input_ids(ids)
        allowed = vocab.ids_matching_prefix(candidates, generated)
        if not allowed:
            break
        next_id = _masked_argmax(logits, allowed)
        if next_id is None:
            break
        ids.append(next_id)
        generated += vocab.clean(vocab.id_to_token[next_id])

    if generated not in candidates:
        # Safety net: mask construction guarantees this should not happen,
        # but never let a partial match crash the batch.
        matches = [c for c in candidates if c.startswith(generated)]
        generated = matches[0] if matches else candidates[0]
    return generated


def _extract_number(
    model: "Small_LLM_Model", vocab: Vocabulary, input_ids: list[int]
) -> float:
    ids = list(input_ids)
    digits = ""
    for i in range(MAX_NUMBER_TOKENS):
        logits = model.get_logits_from_input_ids(ids)
        # Check whether the model's own (unmasked) top choice is still
        # numeric-looking; once it drifts away and we already have digits,
        # treat that as the natural end of the number.
        unmasked_id = _argmax(logits)
        unmasked_str = vocab.clean(vocab.id_to_token.get(unmasked_id, ""))
        if digits and not _looks_numeric(unmasked_str):
            break
        allowed = set(vocab.digit_token_ids) | set(vocab.dot_token_ids)
        if i == 0:
            allowed |= vocab.minus_token_ids
        next_id = _masked_argmax(logits, allowed)
        if next_id is None:
            break
        digits += vocab.clean(vocab.id_to_token[next_id])
        ids.append(next_id)
    try:
        return float(digits)
    except ValueError:
        return 0.0


def _extract_string(
    model: "Small_LLM_Model", vocab: Vocabulary, input_ids: list[int]
) -> str:
    ids = list(input_ids)
    text = ""
    for _ in range(MAX_STRING_TOKENS):
        logits = model.get_logits_from_input_ids(ids)
        next_id = _argmax(logits)
        token_str = vocab.clean(vocab.id_to_token.get(next_id, ""))
        if token_str in _STOP_STRINGS or token_str == "":
            break
        text += token_str
        ids.append(next_id)
    # json.dumps (used when the result is written) escapes any special
    # character in this string automatically, so JSON validity does not
    # depend on what the model produced here.
    return text.strip().strip('"').strip("'")


def _extract_boolean(
    model: "Small_LLM_Model", vocab: Vocabulary, input_ids: list[int]
) -> bool:
    value = _select_from_candidates(
        model, vocab, input_ids, ["true", "false"], MAX_BOOL_TOKENS
    )
    return value == "true"


def _build_selection_prompt(user_prompt: str, functions: list[FunctionDef]) -> str:
    lines = [f"- {fn.name}: {fn.description}" for fn in functions]
    return (
        "You are a function-routing assistant. Choose exactly one function "
        "name that best answers the user request below. Reply with only "
        "the function name, nothing else.\n\n"
        "Available functions:\n" + "\n".join(lines) + "\n\n"
        f'User request: "{user_prompt}"\n'
        "Function name:"
    )


def _build_field_prompt(user_prompt: str, fn: FunctionDef, param_name: str) -> str:
    return (
        f'User request: "{user_prompt}"\n'
        f"Function selected: {fn.name} - {fn.description}\n"
        f"Give the exact value for parameter '{param_name}' and nothing "
        "else, on a single line.\n"
        "Value:"
    )


def _encode(model: "Small_LLM_Model", text: str) -> list[int]:
    ids: Any = model.encode(text)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def generate_function_call(
    user_prompt: str,
    functions: list[FunctionDef],
    model: "Small_LLM_Model",
    vocab: Vocabulary,
) -> FunctionCallResult:
    selection_prompt = _build_selection_prompt(user_prompt, functions)
    selection_ids = _encode(model, selection_prompt)
    names = [fn.name for fn in functions]
    chosen_name = _select_from_candidates(
        model, vocab, selection_ids, names, MAX_NAME_TOKENS
    )
    fn = next(f for f in functions if f.name == chosen_name)

    parameters: dict[str, ParamValue] = {}
    for param_name, schema in fn.parameters.items():
        field_prompt = _build_field_prompt(user_prompt, fn, param_name)
        field_ids = _encode(model, field_prompt)
        if schema.type == "number":
            parameters[param_name] = _extract_number(model, vocab, field_ids)
        elif schema.type == "boolean":
            parameters[param_name] = _extract_boolean(model, vocab, field_ids)
        else:
            parameters[param_name] = _extract_string(model, vocab, field_ids)

    return FunctionCallResult(prompt=user_prompt, name=fn.name, parameters=parameters)
