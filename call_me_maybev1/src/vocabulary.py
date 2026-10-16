# ABOUTME: Loads the model's vocabulary file and builds token<->id lookup maps.
# ABOUTME: This is the foundation constrained decoding uses to inspect what each token ID means.

import json
from pathlib import Path
from typing import Protocol


class VocabFileNotFoundError(Exception):
    """Raised when the vocabulary file cannot be found or parsed."""


class TokenProvider(Protocol):
    """Minimal protocol describing the part of Small_LLM_Model we depend on here.

    Using a Protocol (instead of importing llm_sdk's concrete class) keeps this
    module decoupled and easy to test with a fake model.
    """

    def get_path_to_vocab_file(self) -> str: ...


class Vocabulary:
    """Bidirectional mapping between token strings and their integer IDs.

    Attributes:
        token_to_id: Maps a token's raw string (as stored by the tokenizer,
            e.g. "Ġworld") to its integer ID.
        id_to_token: The reverse mapping, integer ID to token string.
    """

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id: dict[str, int] = token_to_id
        self.id_to_token: dict[int, str] = {
            token_id: token for token, token_id in token_to_id.items()
        }

    @classmethod
    def from_file(cls, vocab_file_path: Path) -> "Vocabulary":
        """Build a Vocabulary by reading a vocab.json-style file.

        Args:
            vocab_file_path: Path to the tokenizer's vocabulary file, as
                returned by Small_LLM_Model.get_path_to_vocab_file().

        Returns:
            A populated Vocabulary instance.

        Raises:
            VocabFileNotFoundError: If the file is missing, unreadable, or
                does not contain a JSON object of {token: id} pairs.
        """
        try:
            with vocab_file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise VocabFileNotFoundError(
                f"Vocabulary file not found: {vocab_file_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise VocabFileNotFoundError(
                f"Vocabulary file is not valid JSON: {vocab_file_path} ({exc})"
            ) from exc

        if not isinstance(data, dict):
            raise VocabFileNotFoundError(
                f"Expected a JSON object of token->id pairs in {vocab_file_path}, "
                f"got {type(data).__name__}"
            )
        return cls(token_to_id=data)

    def token_for_id(self, token_id: int) -> str | None:
        """Return the raw token string for a given ID, or None if unknown."""
        return self.id_to_token.get(token_id)

    def id_for_token(self, token: str) -> int | None:
        """Return the ID for a given raw token string, or None if unknown."""
        return self.token_to_id.get(token)

    def __len__(self) -> int:
        return len(self.token_to_id)


def load_vocabulary(model: TokenProvider) -> Vocabulary:
    """Load the Vocabulary for a given LLM wrapper.

    Args:
        model: An object exposing get_path_to_vocab_file(), such as
            llm_sdk.Small_LLM_Model.

    Returns:
        The loaded Vocabulary.

    Raises:
        VocabFileNotFoundError: If the vocabulary file cannot be found or parsed.
    """
    vocab_path = Path(model.get_path_to_vocab_file())
    return Vocabulary.from_file(vocab_path)