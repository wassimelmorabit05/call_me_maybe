# ABOUTME: Loads the tokenizer vocabulary (token string <-> token id) and
# ABOUTME: builds pre-computed index sets used by constrained decoding.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# BPE tokenizers mark a token that was preceded by a space with a special
# character. Qwen/GPT-style tokenizers commonly use "\u0120" (a "Ġ" glyph);
# SentencePiece-style tokenizers use "\u2581" ("▁"). We normalize both to a
# literal leading space so we can reason about plain text.
_SPACE_MARKERS = ("\u0120", "\u2581")


@dataclass
class Vocabulary:
    """Wraps the model's vocabulary file and exposes fast lookups needed by
    constrained decoding: which token ids are pure digits, a lone ".", a
    lone "-", or made only of letters."""

    id_to_token: dict[int, str]
    token_to_id: dict[str, int]

    digit_token_ids: set[int] = field(default_factory=set)
    letter_token_ids: set[int] = field(default_factory=set)
    dot_token_ids: set[int] = field(default_factory=set)
    minus_token_ids: set[int] = field(default_factory=set)

    @classmethod
    def load(cls, vocab_path: str) -> "Vocabulary":
        with Path(vocab_path).open("r", encoding="utf-8") as f:
            token_to_id: dict[str, int] = json.load(f)
        id_to_token = {tid: tok for tok, tid in token_to_id.items()}
        vocab = cls(id_to_token=id_to_token, token_to_id=token_to_id)
        vocab._build_indexes()
        return vocab

    @staticmethod
    def clean(token_str: str) -> str:
        """Replace the tokenizer's leading-space marker with a real space."""
        for marker in _SPACE_MARKERS:
            token_str = token_str.replace(marker, " ")
        return token_str

    def _build_indexes(self) -> None:
        for tid, tok in self.id_to_token.items():
            stripped = self.clean(tok).strip()
            if not stripped:
                continue
            if stripped.isdigit():
                self.digit_token_ids.add(tid)
            elif stripped.isalpha():
                self.letter_token_ids.add(tid)
            elif stripped == ".":
                self.dot_token_ids.add(tid)
            elif stripped == "-":
                self.minus_token_ids.add(tid)

    def ids_matching_prefix(self, candidates: list[str], so_far: str) -> set[int]:
        """Return every token id whose (cleaned, left-stripped when `so_far`
        is empty) string could be appended to `so_far` while remaining a
        prefix of at least one string in `candidates`."""
        remaining = [c for c in candidates if c.startswith(so_far)]
        if not remaining:
            return set()
        valid: set[int] = set()
        for tid, tok in self.id_to_token.items():
            cleaned = self.clean(tok)
            piece = cleaned.lstrip() if so_far == "" else cleaned
            if not piece:
                continue
            for option in remaining:
                rest = option[len(so_far):]
                if rest.startswith(piece):
                    valid.add(tid)
                    break
        return valid
