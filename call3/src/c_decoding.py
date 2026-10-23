import json
import numpy as np
from numpy import ndarray


def load_vocabulary(
    vocab_path: str
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Load the model's vocabulary from a JSON file.

    Create two dictionaries mapping tokens to their IDs and IDs to
    their corresponding tokens.

    Return:
        A tuple containing both vocabulary dictionaries.
    """
    try:
        with open(vocab_path, "r", encoding="utf-8") as file:
            token_to_id = json.load(file)
        id_to_token = {v: k for k, v in token_to_id.items()}
        return token_to_id, id_to_token
    except Exception as e:
        print(e)
        raise


def get_valid_next_tokens(
    current_prefix: str,
    function_names: list[str],
    token_to_id: dict[str, int]
) -> set[int]:
    """
    Find token IDs that can continue the current function name prefix.

    Check each vocabulary token and keep its ID if appending it to the
    current prefix still matches the beginning of an available function
    name.

    Return:
        A set of valid next-token IDs.
    """
    valid_strings = [
        func_name
        for func_name in function_names
        if func_name.startswith(current_prefix)
    ]
    valid_tokens = set()
    for token, id in token_to_id.items():
        new_prefix = current_prefix + token

        for s in valid_strings:
            if s.startswith(new_prefix):
                valid_tokens.add(id)
                break
    return valid_tokens


def mask_logits(
    logits: ndarray,
    valid_token_ids: set[int]
) -> ndarray:
    """
    Restrict token selection to the provided valid token IDs.

    Create an array filled with negative infinity and restore the
    original logits only at valid token positions.

    Return:
        A NumPy array containing the masked logits.
    """
    logits = np.array(logits)

    masked_logits = np.full_like(logits, -np.inf)
    valid_ids = list(valid_token_ids)
    masked_logits[valid_ids] = logits[valid_ids]
    return masked_logits
