import json

from .models import FunctionCallingTest, FunctionDefinition


def load_json(path: str) -> object:
    """Load data from a JSON file."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_tests(path: str) -> list[FunctionCallingTest]:
    """Load and validate function calling tests."""

    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(
            "Function calling tests must be a JSON array."
        )

    tests = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError(
                "Each function calling test must be an object."
            )

        tests.append(FunctionCallingTest(**item))

    return tests


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions."""

    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(
            "Function definitions must be a JSON array."
        )

    functions = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError(
                "Each function definition must be an object."
            )

        functions.append(FunctionDefinition(**item))

    return functions
