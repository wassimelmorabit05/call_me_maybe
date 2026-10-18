# ABOUTME: Loads the model's vocabulary file and builds token<->id lookup maps.
# ABOUTME: This is the foundation constrained decoding uses to inspect what each token ID means.
#
# WHY THIS FILE EXISTS (read this first):
# The model only ever talks in numbers (token IDs). To decide whether a
# given ID is "allowed" by our JSON rules, we first need to know WHAT TEXT
# that ID actually represents. This file loads that mapping once, from the
# tokenizer's own vocabulary file.

import json
from pathlib import Path
from typing import Protocol


class VocabFileNotFoundError(Exception):
    """Raised when the vocabulary file cannot be found or parsed."""


class TokenProvider(Protocol):
    """The only model ability this file needs: where is the vocab file?

    Using a Protocol here (instead of importing llm_sdk.Small_LLM_Model
    directly) keeps this module independent of the real model, so it can be
    tested with a tiny fake object instead.
    """

    def get_path_to_vocab_file(self) -> str: ...


class Vocabulary:
    """A two-way lookup between a token's text and its integer ID.

    Attributes:
        token_to_id: text -> id (this is exactly what the vocab file gives us).
        id_to_token: id -> text (the reverse, which we build ourselves,
            since the rest of the program mostly needs to go THIS direction:
            "I have an ID from the logits vector, what text is it?").
    """

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id: dict[str, int] = token_to_id
        self.id_to_token: dict[int, str] = {
            token_id: token for token, token_id in token_to_id.items()
        }

    @classmethod
    def from_file(cls, vocab_file_path: Path) -> "Vocabulary":
        """Read a vocab.json-style file ({"token text": id, ...}) and build
        a Vocabulary from it.

        Args:
            vocab_file_path: Path returned by
                Small_LLM_Model.get_path_to_vocab_file().

        Returns:
            A populated Vocabulary.

        Raises:
            VocabFileNotFoundError: If the file is missing, not valid JSON,
                or isn't a {token: id} object as expected.
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
        """What text does this ID represent? None if we've never seen it."""
        return self.id_to_token.get(token_id)

    def id_for_token(self, token: str) -> int | None:
        """What ID does this exact text correspond to? None if unknown."""
        return self.token_to_id.get(token)

    def __len__(self) -> int:
        return len(self.token_to_id)


def load_vocabulary(model: TokenProvider) -> Vocabulary:
    """Convenience function: ask the model where its vocab file is, and
    load it into a Vocabulary in one call.

    Args:
        model: Anything with a get_path_to_vocab_file() method, such as
            llm_sdk.Small_LLM_Model.

    Returns:
        The loaded Vocabulary.

    Raises:
        VocabFileNotFoundError: If the vocabulary file can't be found/parsed.
    """
    vocab_path = Path(model.get_path_to_vocab_file())
    return Vocabulary.from_file(vocab_path)