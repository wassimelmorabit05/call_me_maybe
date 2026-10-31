# ABOUTME: Vocabulary loading and the generic constrained-decoding primitives:
# ABOUTME: masking invalid tokens to -inf and picking the best surviving one.

import json
from typing import Any

import numpy as np
from numpy import ndarray


def load_vocabulary(vocab_path: str) -> dict[str, int]:
    """Load the raw token->id vocabulary mapping from the tokenizer's vocab file."""
    with open(vocab_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_token_text_cache(model: Any, token_to_id: dict[str, int]) -> dict[int, str]:
    """Pre-decode every vocabulary token ID into its human-readable text, once.

    Done once per run (not once per generation step), and via model.decode()
    rather than the raw vocab.json keys, so BPE's internal space marker
    ("Ġ") is already turned back into a literal space - no manual
    replacement needed anywhere else in the code.
    """
    cache: dict[int, str] = {}
    for token_id in token_to_id.values():
        text = model.decode([token_id])
        if text:
            cache[token_id] = text
    return cache


def get_valid_next_tokens(
    current_prefix: str,
    choices: list[str],
    token_text_cache: dict[int, str],
) -> set[int]:
    """Find token IDs that keep `current_prefix` a valid prefix of a known choice.

    Used both for the function name (choices = known function names) and for
    boolean values (choices = ["true", "false"]).
    """
    candidates = [choice for choice in choices if choice.startswith(current_prefix)]
    valid_tokens: set[int] = set()
    for token_id, text in token_text_cache.items():
        new_prefix = current_prefix + text
        if any(choice.startswith(new_prefix) for choice in candidates):
            valid_tokens.add(token_id)
    return valid_tokens


def mask_logits(logits: Any, valid_token_ids: set[int]) -> ndarray:
    """Set every logit whose token ID is not in valid_token_ids to -infinity.

    Args:
        logits: The raw logits vector from get_logits_from_input_ids().
        valid_token_ids: IDs allowed to be selected next.

    Returns:
        A new numpy array, same length as the input, with invalid entries
        set to -infinity so they can never win argmax.
    """
    logits_array = np.asarray(logits, dtype=float)
    masked_logits = np.full_like(logits_array, -np.inf)
    ids = list(valid_token_ids)
    masked_logits[ids] = logits_array[ids]
    return masked_logits


def select_next_token(masked_logits: ndarray) -> int:
    """Greedy decoding: return the ID of the highest-scoring surviving token.

    Raises:
        ValueError: If every entry is -infinity (no valid token at all).
    """
    next_id = int(np.argmax(masked_logits))
    if not np.isfinite(masked_logits[next_id]):
        raise ValueError("No valid token available for the current decoding state")
    return next_id