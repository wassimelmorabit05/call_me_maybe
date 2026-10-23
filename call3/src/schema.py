from pydantic import BaseModel


class FunctionCallResult(BaseModel):
    """Model for validating function call results.

    Attributes:
        prompt: Original user prompt.
        name: Name of the function to call.
        parameters: Dictionary of parameter names and their values.
    """
    prompt: str
    name: str
    parameters: dict[str, bool | str | float | int]
