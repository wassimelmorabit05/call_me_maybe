# ABOUTME: Pydantic models for functions_definition.json, function_calling_tests.json
# ABOUTME: and the output schema (function_calling_results.json).

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel

ParamType = Literal["number", "string", "boolean"]


class ParamSchema(BaseModel):
    """Type descriptor for a single function parameter or return value."""

    model_config = ConfigDict(extra="forbid")

    type: ParamType


class FunctionDef(BaseModel):
    """One entry of functions_definition.json."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParamSchema] = Field(default_factory=dict)
    returns: ParamSchema


class FunctionsDefinition(RootModel[list[FunctionDef]]):
    """The full functions_definition.json array."""


class PromptItem(BaseModel):
    """One entry of function_calling_tests.json."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


class PromptsFile(RootModel[list[PromptItem]]):
    """The full function_calling_tests.json array."""


ParamValue = Union[float, str, bool]


class FunctionCallResult(BaseModel):
    """One entry written to data/output/function_calling_results.json."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, ParamValue]
