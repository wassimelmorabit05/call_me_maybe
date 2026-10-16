# ABOUTME: Unit tests for DecodingState classes (ChoiceState, NumberState, StringState).
# ABOUTME: Pure logic tests, no model or vocabulary required.

from src.decoding_states import ChoiceState, NumberState, StringState


class TestChoiceState:
    def test_valid_prefix_is_accepted(self) -> None:
        state = ChoiceState(["fn_add_numbers", "fn_greet"])
        assert state.is_valid_addition("fn_") is True
        assert state.is_valid_addition("fn_add") is True

    def test_invalid_prefix_is_rejected(self) -> None:
        state = ChoiceState(["fn_add_numbers", "fn_greet"])
        assert state.is_valid_addition("xyz") is False

    def test_cannot_stop_on_partial_match(self) -> None:
        state = ChoiceState(["fn_add_numbers"])
        state.accept("fn_add")
        assert state.can_stop() is False

    def test_can_stop_on_exact_match(self) -> None:
        state = ChoiceState(["fn_greet"])
        state.accept("fn_greet")
        assert state.can_stop() is True

    def test_requires_at_least_one_choice(self) -> None:
        try:
            ChoiceState([])
        except ValueError:
            return
        raise AssertionError("Expected ValueError for empty choices")


class TestNumberState:
    def test_digits_are_valid(self) -> None:
        state = NumberState()
        assert state.is_valid_addition("2") is True
        assert state.is_valid_addition("42") is True

    def test_leading_minus_is_valid(self) -> None:
        state = NumberState()
        assert state.is_valid_addition("-") is True

    def test_minus_not_at_start_is_invalid(self) -> None:
        state = NumberState()
        state.accept("3")
        assert state.is_valid_addition("-") is False

    def test_single_decimal_point_allowed(self) -> None:
        state = NumberState()
        state.accept("3")
        assert state.is_valid_addition(".") is True
        state.accept(".")
        assert state.is_valid_addition(".") is False

    def test_non_digit_is_invalid(self) -> None:
        state = NumberState()
        assert state.is_valid_addition("a") is False

    def test_cannot_stop_when_empty_or_just_minus(self) -> None:
        state = NumberState()
        assert state.can_stop() is False
        state.accept("-")
        assert state.can_stop() is False

    def test_cannot_stop_on_trailing_dot(self) -> None:
        state = NumberState()
        state.accept("3")
        state.accept(".")
        assert state.can_stop() is False

    def test_can_stop_on_complete_integer(self) -> None:
        state = NumberState()
        state.accept("265")
        assert state.can_stop() is True

    def test_can_stop_on_complete_decimal(self) -> None:
        state = NumberState()
        state.accept("3")
        state.accept(".")
        state.accept("5")
        assert state.can_stop() is True


class TestStringState:
    def test_quote_is_invalid(self) -> None:
        state = StringState()
        assert state.is_valid_addition('"') is False

    def test_backslash_is_invalid(self) -> None:
        state = StringState()
        assert state.is_valid_addition("\\") is False

    def test_plain_text_is_valid(self) -> None:
        state = StringState()
        assert state.is_valid_addition("hello") is True

    def test_empty_string_can_stop(self) -> None:
        state = StringState()
        assert state.can_stop() is True

    def test_non_empty_string_can_stop(self) -> None:
        state = StringState()
        state.accept("hello")
        assert state.can_stop() is True