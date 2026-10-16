# ABOUTME: Functions to load and validate the project's JSON input files.
# ABOUTME: Wraps file I/O, JSON parsing, and pydantic validation into clear, catchable errors.

import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionDefinition, TestPrompt


class InputFileError(Exception):
    """Raised when an input file is missing, not valid JSON, or does not match
    the expected schema. Carries a human-readable message safe to show the user.
    """


def _read_json_array(path: Path) -> list[dict]:
    """Read a file and parse it as a JSON array.

    Args:
        path: Path to the JSON file to read.

    Returns:
        The parsed JSON content as a list of dictionaries.

    Raises:
        InputFileError: If the file is missing, unreadable, not valid JSON,
            or its top-level structure is not a JSON array.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise InputFileError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputFileError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise InputFileError(f"Could not read {path}: {exc}") from exc

    if not isinstance(data, list):
        raise InputFileError(
            f"Expected a JSON array at the top level of {path}, "
            f"got {type(data).__name__}"
        )
    return data


def load_functions_definition(path: Path) -> list[FunctionDefinition]:
    """Load and validate the function definitions file.

    Args:
        path: Path to functions_definition.json.

    Returns:
        A list of validated FunctionDefinition objects.

    Raises:
        InputFileError: If the file is missing, invalid JSON, or does not
            match the expected schema.
    """
    raw_items = _read_json_array(path)
    try:
        return [FunctionDefinition.model_validate(item) for item in raw_items]
    except ValidationError as exc:
        raise InputFileError(
            f"{path} does not match the expected function definition schema: {exc}"
        ) from exc


def load_test_prompts(path: Path) -> list[TestPrompt]:
    """Load and validate the test prompts file.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        A list of validated TestPrompt objects.

    Raises:
        InputFileError: If the file is missing, invalid JSON, or does not
            match the expected schema.
    """
    raw_items = _read_json_array(path)
    try:
        return [TestPrompt.model_validate(item) for item in raw_items]
    except ValidationError as exc:
        raise InputFileError(
            f"{path} does not match the expected test prompt schema: {exc}"
        ) from exc