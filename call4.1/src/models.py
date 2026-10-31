# ABOUTME: Pydantic model for the final, validated function-call result.
# ABOUTME: This is the exact shape written to the output JSON file.

from pydantic import BaseModel


class FunctionCallResult(BaseModel):
    """A single translated function call.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the function to call.
        parameters: The generated argument values, keyed by parameter name.
    """

    prompt: str
    name: str
    parameters: dict[str, bool | str | float | int]