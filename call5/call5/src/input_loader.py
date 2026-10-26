"""Read JSON files and report clear input errors."""

import json

from pydantic import ValidationError

from src.models import FunctionDefinition, Prompt


def read_json(path: str) -> list:
    """Read a JSON array from a UTF-8 file."""
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def validate_prompts(path: str) -> list:
    """Load and validate each prompt, preserving its order."""
    prompts = []
    for index, data in enumerate(read_json(path), 1):
        try:
            prompts.append(Prompt.model_validate(data))
        except ValidationError as error:
            message = error.errors()[0]["msg"]
            raise ValueError(f"Prompt {index}: {message}") from error
    return prompts


def validate_functions(path: str) -> list:
    """Load function definitions and reject duplicate names."""
    functions = []
    names: set = set()
    for index, data in enumerate(read_json(path), 1):
        try:
            function = FunctionDefinition.model_validate(data)
        except ValidationError as error:
            message = error.errors()[0]["msg"]
            raise ValueError(f"Function {index}: {message}") from error
        if function.name in names:
            raise ValueError(f"Duplicate function: {function.name}")
        names.add(function.name)
        functions.append(function)
    if not functions:
        raise ValueError("The function definitions array is empty")
    return functions
