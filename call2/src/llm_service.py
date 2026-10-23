from llm_sdk import Small_LLM_Model

from .decoder import (
    get_valid_next_tokens,
    select_best_valid_token,
)
from .models import FunctionDefinition
from .prompts import build_function_selection_prompt
from typing import cast


def create_model() -> Small_LLM_Model:
    """Create and return the language model."""
    model = Small_LLM_Model()
    return model


def select_function(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Select the most appropriate function using the LLM."""

    prompt = build_function_selection_prompt(
        user_prompt,
        functions,
    )

    prompt_tensor = model.encode(prompt)
    prompt_ids = prompt_tensor[0].tolist()

    candidates = []

    for function in functions:
        encoded = model.encode(function.name)
        token_ids = encoded[0].tolist()
        candidates.append(token_ids)

    generated_tokens: list[int] = []

    while True:
        valid_tokens = get_valid_next_tokens(
            candidates,
            generated_tokens,
        )

        if not valid_tokens:
            break

        current_input_ids = prompt_ids + generated_tokens

        logits = model.get_logits_from_input_ids(
            current_input_ids
        )

        next_token = select_best_valid_token(
            logits,
            valid_tokens,
        )

        generated_tokens.append(next_token)

        if generated_tokens in candidates:
            break

    selected_function = cast(
        str,
        model.decode(generated_tokens),
    )

    return selected_function
