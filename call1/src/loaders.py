# ABOUTME: Functions to load and validate the project's JSON input files.
# ABOUTME: Wraps file I/O, JSON parsing, and pydantic validation into clear, catchable errors.
#
# WHY THIS FILE EXISTS (read this first):
# The subject requires that bad input files (missing, broken JSON, wrong
# shape) never crash the program and always give a clear message. Rather
# than scattering try/except everywhere, every possible failure in reading
# EITHER input file funnels into ONE exception type: InputFileError. The
# caller (src/__main__.py) only has to catch that one thing.

import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionDefinition, TestPrompt


class InputFileError(Exception):
    """Any problem with an input file: missing, invalid JSON, or wrong shape.

    The message is written to be shown directly to a user (no internal
    jargon), since __main__.py just prints str(exc) as-is.
    """


def _read_json_array(path: Path) -> list[dict]:
    """Read a file and parse it, expecting a JSON array ([...]) at the top.

    Both of this project's input files are JSON arrays of objects, so this
    one helper is shared by both load_functions_definition() and
    load_test_prompts() below.
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
    """Load and validate functions_definition.json.

    Args:
        path: Path to the file.

    Returns:
        One FunctionDefinition (a validated pydantic object) per entry.

    Raises:
        InputFileError: File missing/unreadable, invalid JSON, or an entry
            doesn't match the expected schema (e.g. a missing field).
    """
    raw_items = _read_json_array(path)
    try:
        # model_validate() is pydantic's way of turning a plain dict into a
        # checked object - it raises ValidationError with a clear message
        # if anything doesn't match (wrong type, missing field, etc.).
        return [FunctionDefinition.model_validate(item) for item in raw_items]
    except ValidationError as exc:
        raise InputFileError(
            f"{path} does not match the expected function definition schema: {exc}"
        ) from exc


def load_test_prompts(path: Path) -> list[TestPrompt]:
    """Load and validate function_calling_tests.json.

    Args:
        path: Path to the file.

    Returns:
        One TestPrompt per entry.

    Raises:
        InputFileError: File missing/unreadable, invalid JSON, or an entry
            doesn't match the expected schema.
    """
    raw_items = _read_json_array(path)
    try:
        return [TestPrompt.model_validate(item) for item in raw_items]
    except ValidationError as exc:
        raise InputFileError(
            f"{path} does not match the expected test prompt schema: {exc}"
        ) from exc