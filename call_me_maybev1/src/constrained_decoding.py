# ABOUTME: Turns a DecodingState into an actual logits mask, and picks the next token.
# ABOUTME: This is the point where "which tokens are valid" meets "what the model actually scored".

from typing import Protocol

from src.decoding_states import DecodingState
from src.vocabulary import Vocabulary

NEG_INF = float("-inf")


class DecoderModel(Protocol):
    """Minimal protocol for the model methods constrained decoding depends on."""

    def decode(self, token_ids: list[int]) -> str: ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: ...


def build_token_text_cache(
    model: DecoderModel, vocabulary: Vocabulary
) -> dict[int, str]:
    """Pre-decode every token ID in the vocabulary to its human-readable text.

    This is done ONCE per run (not once per generation step) because a
    token's decoded text never changes. Reusing this cache at every step is
    what keeps constrained decoding fast despite checking every vocabulary
    entry at every step.

    Args:
        model: The LLM wrapper, used only for its decode() method here.
        vocabulary: The loaded Vocabulary, used to know which IDs exist.

    Returns:
        Mapping of token ID -> its decoded text (empty string entries, e.g.
        for special/control tokens, are kept out of the result).
    """
    cache: dict[int, str] = {}
    for token_id in vocabulary.id_to_token:
        text = model.decode([token_id])
        if text:
            cache[token_id] = text
    return cache


def compute_valid_token_ids(
    state: DecodingState, token_text_cache: dict[int, str]
) -> set[int]:
    """Determine which token IDs are valid next choices for the given state.

    Args:
        state: The current DecodingState for the blank being generated.
        token_text_cache: Precomputed {token_id: decoded_text}, from
            build_token_text_cache().

    Returns:
        The set of token IDs that would keep the blank valid if chosen next.
    """
    return {
        token_id
        for token_id, text in token_text_cache.items()
        if state.is_valid_addition(text)
    }


def mask_logits(logits: list[float], valid_token_ids: set[int]) -> list[float]:
    """Set every logit whose token ID is not in valid_token_ids to -infinity.

    Args:
        logits: The raw logits vector from get_logits_from_input_ids().
        valid_token_ids: IDs allowed to be selected next.

    Returns:
        A new logits list, same length as the input, with invalid entries
        replaced by -infinity so they can never win argmax.
    """
    return [
        value if token_id in valid_token_ids else NEG_INF
        for token_id, value in enumerate(logits)
    ]


def select_next_token(masked_logits: list[float]) -> int:
    """Greedy decoding: return the ID of the highest-scoring token.

    Args:
        masked_logits: Logits after mask_logits() has been applied.

    Returns:
        The token ID with the highest score.

    Raises:
        ValueError: If every entry is -infinity (no valid token at all),
            which signals a bug in the calling state/schema logic.
    """
    best_id = max(range(len(masked_logits)), key=lambda i: masked_logits[i])
    if masked_logits[best_id] == NEG_INF:
        raise ValueError("No valid token available for the current decoding state")
    return best_id