from pydantic import BaseModel


class FunctionCallingTest(BaseModel):
    """Represent a single function calling test."""

    prompt: str


class TypeDefinition(BaseModel):
    """Represent a type definition."""

    type: str


class FunctionDefinition(BaseModel):
    """Represent a function definition."""

    name: str
    description: str
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition


class FunctionCallResult(BaseModel):
    """Represent a generated function call result."""

    prompt: str
    name: str
    parameters: dict[str, float | str]
