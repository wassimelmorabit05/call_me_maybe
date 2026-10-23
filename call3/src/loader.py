"""Read and validate call_me_maybe input files."""

import sys
import json
import keyword
from typing import Dict, Literal, Any, NoReturn

from pydantic import (
    Field,
    BaseModel,
    ConfigDict,
    ValidationError,
    model_validator
)


JsonType = Literal["string", "number", "integer", "boolean"]


class Prompt(BaseModel):
    """Validate a user prompt and reject empty or unexpected fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1)


class Parameter(BaseModel):
    """Validate the declared JSON type of a function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class FunctionDefinition(BaseModel):
    """
    Validate a function definition and its required fields.

    Store the function name, description, parameter definitions,
    and return type.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Parameter] = Field(min_length=1)
    returns: Parameter

    @model_validator(mode="after")
    def validate_name(self) -> 'FunctionDefinition':
        """Validate that the function name is a valid Python identifier."""
        if not self.name.isidentifier() or keyword.iskeyword(self.name):
            raise ValueError(f"Invalid function name: '{self.name}'")
        return self


def read_json(path: str) -> Any:
    """
    Read and parse a JSON file from the given path.

    Return:
        The parsed JSON data.

    Exit the program if the file cannot be read or contains invalid JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        sys.exit(f"[ERROR]: File '{path}' not found")

    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR]: Invalid JSON in '{path}': {e}")

    except OSError as e:
        # FileNotFoundError + PermissionError + IsADirectoryError
        sys.exit(f"[ERROR]: Cannot read '{path}': {e}")


def exit_with_validation_error(
    e: ValidationError,
    location: str
) -> NoReturn:
    """
    Display the first Pydantic validation error and exit the program.

    Include the provided location to identify where validation failed.
    """
    error = e.errors()[0]
    msg = error["msg"]

    sys.exit(f"[ERROR] in {location}: {msg}")


def validate_prompts(prompts_path: str) -> list[Prompt]:
    """
    Load and validate prompts from a JSON file.

    Ensure the input is a list and validate each entry using
    the Prompt model. Exit the program if validation fails.

    Return:
        A list of validated Prompt objects.
    """
    prompts = []
    data = read_json(prompts_path)

    if not isinstance(data, list):
        sys.exit("[ERROR]: Prompts must be a list")

    if not data:
        sys.exit("[ERROR]: Prompts list cannot be empty")

    for i, p in enumerate(data, 1):
        try:
            prompts.append(Prompt.model_validate(p))
        except ValidationError as e:
            exit_with_validation_error(e, f"prompt {i}")

    return prompts


def validate_functions(
    functions_definition_path: str
) -> list[FunctionDefinition]:
    """
    Load and validate function definitions from a JSON file.

    Ensure the input is a nonempty list, validate each function using
    the FunctionDefinition model, and reject duplicate function names.

    Exit the program if validation fails.

    Return:
        A list of validated FunctionDefinition objects.
    """
    functions = []
    data = read_json(functions_definition_path)

    if not isinstance(data, list):
        sys.exit("[ERROR]: Functions must be a list")

    if not data:
        sys.exit("[ERROR]: Functions list cannot be empty")

    names = set()

    for i, f in enumerate(data, 1):
        try:
            func = FunctionDefinition.model_validate(f)
        except ValidationError as e:
            exit_with_validation_error(e, f"function {i}")

        if func.name in names:
            sys.exit(f"[ERROR]: Duplicate function '{func.name}'")

        names.add(func.name)
        functions.append(func)

    return functions
