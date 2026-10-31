# ABOUTME: Read and validate call_me_maybe input files (functions_definition.json,
# ABOUTME: function_calling_tests.json), exiting with a clear message on any failure.

import json
import keyword
import sys
from typing import Any, Dict, NoReturn

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Prompt(BaseModel):
    """A single natural-language test prompt. Rejects empty or unexpected fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1)


class Parameter(BaseModel):
    """The declared schema of a single function parameter.

    `type` is intentionally kept as a plain string rather than a restricted
    Literal: an unexpected/unsupported type here should fail *that one
    function's* generation later (see generator.py), not abort loading of
    the entire file. The given input files may change during peer review,
    and a single unusual type must not take down the whole run.
    """

    model_config = ConfigDict(extra="forbid")

    type: str


class FunctionDefinition(BaseModel):
    """A single callable function, as described in functions_definition.json."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Parameter] = Field(default_factory=dict)
    returns: Parameter

    @model_validator(mode="after")
    def validate_names(self) -> "FunctionDefinition":
        """Ensure function and parameter names are safe Python identifiers."""
        if not self.name.isidentifier() or keyword.iskeyword(self.name):
            raise ValueError(f"Invalid function name: '{self.name}'")
        for parameter_name in self.parameters:
            if not parameter_name.isidentifier() or keyword.iskeyword(parameter_name):
                raise ValueError(f"Invalid parameter name: '{parameter_name}'")
        return self


def read_json(path: str) -> Any:
    """Read and parse a JSON file.

    Returns:
        The parsed JSON data.

    Exits the program with a clear message if the file is missing,
    unreadable, or not valid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        sys.exit(f"[ERROR]: File '{path}' not found")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR]: Invalid JSON in '{path}': {e}")
    except OSError as e:
        sys.exit(f"[ERROR]: Cannot read '{path}': {e}")


def exit_with_validation_error(e: ValidationError, location: str) -> NoReturn:
    """Print the first pydantic validation error with context, then exit."""
    error = e.errors()[0]
    sys.exit(f"[ERROR] in {location}: {error['msg']}")


def validate_prompts(prompts_path: str) -> list[Prompt]:
    """Load and validate the test prompts file.

    Returns:
        A list of validated Prompt objects.

    Exits the program with a clear message on any validation failure.
    """
    data = read_json(prompts_path)

    if not isinstance(data, list):
        sys.exit(f"[ERROR]: '{prompts_path}' must contain a JSON array")
    if not data:
        sys.exit(f"[ERROR]: '{prompts_path}' is empty")

    prompts: list[Prompt] = []
    for i, p in enumerate(data, 1):
        try:
            prompts.append(Prompt.model_validate(p))
        except ValidationError as e:
            exit_with_validation_error(e, f"prompt {i}")
    return prompts


def validate_functions(functions_definition_path: str) -> list[FunctionDefinition]:
    """Load and validate the function definitions file.

    Rejects duplicate function names, since the model would otherwise have
    no reliable way to disambiguate between two functions sharing a name.

    Returns:
        A list of validated FunctionDefinition objects.

    Exits the program with a clear message on any validation failure.
    """
    data = read_json(functions_definition_path)

    if not isinstance(data, list):
        sys.exit(f"[ERROR]: '{functions_definition_path}' must contain a JSON array")
    if not data:
        sys.exit(f"[ERROR]: '{functions_definition_path}' is empty")

    functions: list[FunctionDefinition] = []
    names: set[str] = set()
    for i, f in enumerate(data, 1):
        try:
            func = FunctionDefinition.model_validate(f)
        except ValidationError as e:
            exit_with_validation_error(e, f"function {i}")
        if func.name in names:
            sys.exit(f"[ERROR]: Duplicate function name '{func.name}'")
        names.add(func.name)
        functions.append(func)
    return functions