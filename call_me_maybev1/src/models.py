# ABOUTME: Pydantic models describing the schema of the project's input files.
# ABOUTME: Covers function definitions (functions_definition.json) and test prompts (function_calling_tests.json).
#
# WHY THIS FILE EXISTS (read this first):
# Instead of working with raw dicts (and risking a typo like item["nmae"]
# crashing somewhere deep in the code), we describe the exact SHAPE we
# expect once, here, as pydantic classes. Every input file gets converted
# into these objects immediately after reading - if anything doesn't match,
# we find out right away with a clear error (see loaders.py).

from pydantic import BaseModel, Field


class ParameterSchema(BaseModel):
    """One function parameter's schema, e.g. {"type": "number"}."""

    type: str


class ReturnsSchema(BaseModel):
    """A function's return type, e.g. {"type": "number"}."""

    type: str


class FunctionDefinition(BaseModel):
    """One entry from functions_definition.json.

    Attributes:
        name: Unique function identifier, e.g. "fn_add_numbers".
        description: Human-readable explanation (used in the prompt we
            show the model).
        parameters: parameter name -> its schema. Note this is a dict
            (keyed by name), matching the JSON shape {"a": {...}, "b": {...}}
            - NOT a list.
        returns: The function's return type (not actually used by our
            generation logic, but kept so the schema matches the input
            file exactly and validation can catch a malformed entry).
    """

    name: str
    description: str
    parameters: dict[str, ParameterSchema] = Field(default_factory=dict)
    returns: ReturnsSchema


class TestPrompt(BaseModel):
    """One entry from function_calling_tests.json, e.g. {"prompt": "..."}."""

    prompt: str