# ABOUTME: Turns a DecodingState into an actual logits mask, and picks the next token.
# ABOUTME: This is the point where "which tokens are valid" meets "what the model actually scored".
#
# WHY THIS FILE EXISTS (read this first):
# The model's output (get_logits_from_input_ids) is just a big list of numbers,
# one per possible token ID. The functions here do exactly the 3 things from
# the "constrained decoding" recipe:
#   1) figure out which token IDs are currently allowed (ask the DecodingState)
#   2) set every DISALLOWED token's score to -infinity, so it can never win
#   3) pick the token with the highest remaining score (argmax)

from typing import Protocol

from src.decoding_states import DecodingState
from src.vocabulary import Vocabulary

NEG_INF = float("-inf")


class DecoderModel(Protocol):
    """The only two model abilities this file needs.

    We depend on this tiny "shape" instead of importing the real
    Small_LLM_Model class directly. That means we can test everything in
    this file with a small fake object instead of the real (slow, network-
    dependent) model.
    """

    def decode(self, token_ids: list[int]) -> str: ...

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: ...


def build_token_text_cache(
    model: DecoderModel, vocabulary: Vocabulary
) -> dict[int, str]:
    """Figure out, ONE TIME, what text every token ID actually decodes to.

    Why do this once instead of on-the-fly? Because a token's text never
    changes — token ID 4771 always means the same string. If we called
    model.decode() for every token at EVERY generation step, we'd redo the
    exact same work thousands of times for nothing. Doing it once here and
    reusing the result (a plain Python dict) is what keeps the program fast.

    Args:
        model: The LLM wrapper, used only for its decode() method here.
        vocabulary: The loaded Vocabulary, used to know which IDs exist.

    Returns:
        {token_id: its decoded text}. Tokens that decode to "" (usually
        special/control tokens, not real text) are left out on purpose.
    """
    cache: dict[int, str] = {}
    for token_id in vocabulary.id_to_token:
        text = model.decode([token_id])
        if text:  # skip empty/control tokens - they're never a useful choice
            cache[token_id] = text
    return cache


def compute_valid_token_ids(
    state: DecodingState, token_text_cache: dict[int, str]
) -> set[int]:
    """Ask the state: "of ALL known tokens, which ones are OK to add right now?"

    This is a simple loop over the cache we built once in
    build_token_text_cache(): for each (id, text) pair, ask
    state.is_valid_addition(text). No model call happens here at all — it's
    pure bookkeeping, which is why it's cheap to call on every single step.

    Args:
        state: The current DecodingState for the blank being generated.
        token_text_cache: Precomputed {token_id: decoded_text}.

    Returns:
        The set of token IDs that are safe to add next without breaking
        the current blank's rule.
    """
    return {
        token_id
        for token_id, text in token_text_cache.items()
        if state.is_valid_addition(text)
    }


def mask_logits(logits: list[float], valid_token_ids: set[int]) -> list[float]:
    """The actual "masking" step: kill every score that isn't allowed.

    This is the heart of constrained decoding. We don't change ANY of the
    allowed scores — we only overwrite the disallowed ones with -infinity,
    which guarantees (not just "makes unlikely") that argmax can never pick
    them, no matter how high their original score was.

    Args:
        logits: The raw logits vector from get_logits_from_input_ids().
        valid_token_ids: IDs allowed to be selected next.

    Returns:
        A new list, same length as `logits`: unchanged where allowed,
        -infinity everywhere else.
    """
    return [
        value if token_id in valid_token_ids else NEG_INF
        for token_id, value in enumerate(logits)
    ]


def select_next_token(masked_logits: list[float]) -> int:
    """Greedy decoding: just return the index of the highest remaining score.

    This is the "argmax" step from the theory. Because every disallowed
    token is already -infinity, whichever token wins here is guaranteed to
    be a valid choice.

    Args:
        masked_logits: Logits after mask_logits() has already been applied.

    Returns:
        The token ID with the highest score.

    Raises:
        ValueError: If literally everything is -infinity, meaning no token
            at all was valid — this points to a bug in the calling state's
            rules, not a normal situation, so we fail loudly instead of
            picking something arbitrary.
    """
    best_id = max(range(len(masked_logits)), key=lambda i: masked_logits[i])
    if masked_logits[best_id] == NEG_INF:
        raise ValueError("No valid token available for the current decoding state")
    return best_id