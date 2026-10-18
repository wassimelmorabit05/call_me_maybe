# ABOUTME: Unit tests for DecodingState classes (ChoiceState, NumberState, StringState).
# ABOUTME: Pure logic tests, no model or vocabulary required.

from src.decoding_states import ChoiceState, IntegerState, NumberState, StringState


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


class TestIntegerState:
    def test_digits_are_valid(self) -> None:
        state = IntegerState()
        assert state.is_valid_addition("4") is True

    def test_decimal_point_is_invalid(self) -> None:
        state = IntegerState()
        state.accept("4")
        assert state.is_valid_addition(".") is False

    def test_cannot_stop_when_empty_or_just_minus(self) -> None:
        state = IntegerState()
        assert state.can_stop() is False
        state.accept("-")
        assert state.can_stop() is False

    def test_can_stop_after_digits(self) -> None:
        state = IntegerState()
        state.accept("42")
        assert state.can_stop() is True

    def test_negative_integer(self) -> None:
        state = IntegerState()
        state.accept("-")
        state.accept("7")
        assert state.can_stop() is True


class TestStringState:
    def test_bare_quote_is_invalid(self) -> None:
        state = StringState()
        assert state.is_valid_addition('"') is False

    def test_plain_text_is_valid(self) -> None:
        state = StringState()
        assert state.is_valid_addition("hello") is True

    def test_single_backslash_is_valid_as_escape_start(self) -> None:
        state = StringState()
        assert state.is_valid_addition("\\") is True

    def test_empty_string_can_stop(self) -> None:
        state = StringState()
        assert state.can_stop() is True

    def test_non_empty_string_can_stop(self) -> None:
        state = StringState()
        state.accept("hello")
        assert state.can_stop() is True

    def test_cannot_stop_mid_escape(self) -> None:
        state = StringState()
        state.accept("\\")
        assert state.can_stop() is False

    def test_only_quote_or_backslash_allowed_right_after_a_backslash(self) -> None:
        state = StringState()
        state.accept("\\")
        assert state.is_valid_addition("n") is False
        assert state.is_valid_addition('"') is True
        assert state.is_valid_addition("\\") is True

    def test_unescape_handles_escaped_quote(self) -> None:
        # Raw generated text for: Say "hello" -> Say \"hello\"
        state = StringState()
        for ch in 'Say \\"hello\\"':
            state.accept(ch)
        assert state.unescape() == 'Say "hello"'

    def test_unescape_handles_literal_backslash_in_regex(self) -> None:
        # Raw generated text for the value \d+ -> \\d+
        state = StringState()
        for ch in "\\\\d+":
            state.accept(ch)
        assert state.unescape() == "\\d+"

    def test_unescape_handles_windows_path(self) -> None:
        # Raw generated text for: C:\Users\john -> C:\\Users\\john
        state = StringState()
        for ch in "C:\\\\Users\\\\john":
            state.accept(ch)
        assert state.unescape() == "C:\\Users\\john"
