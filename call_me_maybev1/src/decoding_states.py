# ABOUTME: DecodingState classes define what counts as a valid next piece of text
# ABOUTME: for a single "blank" in the JSON template (a choice, a number, or a string).

from abc import ABC, abstractmethod


class DecodingState(ABC):
    """A single blank to fill in the JSON template during constrained decoding.

    A state tracks the text accumulated so far for this blank and answers two
    questions at every generation step: can a given piece of text still be
    appended without breaking validity, and is the blank allowed to end now.
    """

    def __init__(self) -> None:
        self.generated_text: str = ""

    @abstractmethod
    def is_valid_addition(self, addition: str) -> bool:
        """Return True if appending `addition` keeps this blank valid so far."""

    @abstractmethod
    def can_stop(self) -> bool:
        """Return True if the blank is complete and may end right now."""

    def accept(self, addition: str) -> None:
        """Commit a chosen piece of text, growing the blank's generated text."""
        self.generated_text += addition


class ChoiceState(DecodingState):
    """Constrains the blank to one of a fixed, known set of full strings.

    Used for the function name (choice among known function names) and for
    boolean values (choice between "true" and "false").
    """

    def __init__(self, choices: list[str]) -> None:
        super().__init__()
        if not choices:
            raise ValueError("ChoiceState requires at least one choice")
        self._choices = choices

    def is_valid_addition(self, addition: str) -> bool:
        candidate = self.generated_text + addition
        return any(choice.startswith(candidate) for choice in self._choices)

    def can_stop(self) -> bool:
        return self.generated_text in self._choices


class NumberState(DecodingState):
    """Constrains the blank to a JSON number: optional leading '-', digits,
    and at most one decimal point.
    """

    def is_valid_addition(self, addition: str) -> bool:
        return self._is_number_prefix(self.generated_text + addition)

    def can_stop(self) -> bool:
        return self._is_complete_number(self.generated_text)

    @staticmethod
    def _is_number_prefix(text: str) -> bool:
        if text in ("", "-"):
            return True
        seen_dot = False
        for index, char in enumerate(text):
            if char == "-":
                if index != 0:
                    return False
            elif char == ".":
                if seen_dot:
                    return False
                seen_dot = True
            elif not char.isdigit():
                return False
        return True

    @staticmethod
    def _is_complete_number(text: str) -> bool:
        if text in ("", "-") or text.endswith("."):
            return False
        try:
            float(text)
        except ValueError:
            return False
        return True


class StringState(DecodingState):
    """Constrains the blank to the *contents* of a JSON string.

    The surrounding double quotes are fixed template text injected by the
    caller, not generated here, so this state only forbids characters that
    would break out of the string (an unescaped quote or a backslash).
    """

    def is_valid_addition(self, addition: str) -> bool:
        return '"' not in addition and "\\" not in addition

    def can_stop(self) -> bool:
        return True