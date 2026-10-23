import pytest

from src.models import FunctionDefinition, TypeDefinition
from src.validator import validate_arguments


def create_test_function() -> FunctionDefinition:
    return FunctionDefinition(
        name="add",
        description="Add two numbers",
        parameters={
            "a": TypeDefinition(type="number"),
            "b": TypeDefinition(type="number"),
        },
        returns=TypeDefinition(type="number"),
    )


def test_valid_arguments() -> None:
    function = create_test_function()

    result = validate_arguments(
        '{"a": 2, "b": 3}',
        function,
    )

    assert result == {
        "a": 2.0,
        "b": 3.0,
    }


def test_missing_argument() -> None:
    function = create_test_function()

    with pytest.raises(ValueError):
        validate_arguments(
            '{"a": 2}',
            function,
        )
