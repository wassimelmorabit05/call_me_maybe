# ABOUTME: Unit tests for constrained_decoding.py using a small fake vocabulary/model,
# ABOUTME: no real LLM weights or network access required.

from src.constrained_decoding import (
    NEG_INF,
    build_token_text_cache,
    compute_valid_token_ids,
    mask_logits,
    select_next_token,
)
from src.decoding_states import NumberState
from src.vocabulary import Vocabulary


class FakeModel:
    """Fake model whose decode() just looks up a fixed id->text table."""

    def __init__(self, id_to_text: dict[int, str]) -> None:
        self._id_to_text = id_to_text

    def decode(self, token_ids: list[int]) -> str:
        return self._id_to_text.get(token_ids[0], "")

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        raise NotImplementedError("Not needed for these tests")


def _make_vocabulary(id_to_text: dict[int, str]) -> Vocabulary:
    token_to_id = {text: id_ for id_, text in id_to_text.items()}
    return Vocabulary(token_to_id)


class TestBuildTokenTextCache:
    def test_cache_matches_decoded_text(self) -> None:
        id_to_text = {0: "0", 1: "1", 2: ",", 3: ""}
        vocabulary = _make_vocabulary({0: "0", 1: "1", 2: ","})
        model = FakeModel(id_to_text)
        cache = build_token_text_cache(model, vocabulary)
        assert cache == {0: "0", 1: "1", 2: ","}

    def test_empty_decoded_text_is_dropped(self) -> None:
        # A control/special token that decodes to "" must not pollute the cache.
        vocabulary = _make_vocabulary({0: "<pad>"})
        model = FakeModel({0: ""})
        cache = build_token_text_cache(model, vocabulary)
        assert cache == {}


class TestComputeValidTokenIds:
    def test_only_valid_additions_are_returned(self) -> None:
        state = NumberState()
        token_text_cache = {0: "2", 1: "a", 2: ",", 3: "-"}
        valid = compute_valid_token_ids(state, token_text_cache)
        assert valid == {0, 2, 3}  # "2" and "-" are number-prefix-valid,
        # "," is not valid as a NUMBER addition (only relevant as a boundary)


class TestMaskLogits:
    def test_invalid_ids_become_negative_infinity(self) -> None:
        logits = [1.0, 5.0, 2.0]
        masked = mask_logits(logits, valid_token_ids={1})
        assert masked == [NEG_INF, 5.0, NEG_INF]

    def test_valid_ids_are_untouched(self) -> None:
        logits = [1.0, 5.0, 2.0]
        masked = mask_logits(logits, valid_token_ids={0, 1, 2})
        assert masked == logits


class TestSelectNextToken:
    def test_picks_highest_scoring_token(self) -> None:
        assert select_next_token([1.0, 5.0, 2.0]) == 1

    def test_ignores_masked_tokens(self) -> None:
        assert select_next_token([NEG_INF, 5.0, NEG_INF]) == 1

    def test_raises_when_nothing_is_valid(self) -> None:
        try:
            select_next_token([NEG_INF, NEG_INF])
        except ValueError:
            return
        raise AssertionError("Expected ValueError when no token is valid")