# ABOUTME: Pydantic models describing the schema of the project's input files.
# ABOUTME: Covers function definitions (functions_definition.json) and test prompts (function_calling_tests.json).

from pydantic import BaseModel, Field


class ParameterSchema(BaseModel):
    """Schema of a single function parameter.

    Attributes:
        type: The JSON-level type expected for this parameter
            (e.g. "number", "string", "boolean").
    """

    type: str


class ReturnsSchema(BaseModel):
    """Schema of a function's return value.

    Attributes:
        type: The JSON-level type returned by the function.
    """

    type: str


class FunctionDefinition(BaseModel):
    """A single callable function as described in functions_definition.json.

    Attributes:
        name: The function's unique identifier (e.g. "fn_add_numbers").
        description: Human-readable explanation of what the function does.
        parameters: Mapping of parameter name -> its schema.
        returns: Schema of the function's return value.
    """

    name: str
    description: str
    parameters: dict[str, ParameterSchema] = Field(default_factory=dict)
    returns: ReturnsSchema


class TestPrompt(BaseModel):
    """A single natural-language prompt as described in function_calling_tests.json.

    Attributes:
        prompt: The natural-language request to translate into a function call.
    """

    prompt: str