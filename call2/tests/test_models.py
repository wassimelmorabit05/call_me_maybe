import pytest
from pydantic import ValidationError

from src.models import FunctionCallingTest


def test_valid_function_calling_test() -> None:
    test = FunctionCallingTest(
        prompt="What is the sum of 2 and 3?"
    )

    assert test.prompt == "What is the sum of 2 and 3?"


def test_missing_prompt() -> None:
    with pytest.raises(ValidationError):
        FunctionCallingTest()  # type: ignore[call-arg]
