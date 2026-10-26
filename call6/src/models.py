"""Validate input definitions and generated values."""

import keyword
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic import model_validator


class Prompt(BaseModel):
    """Store one request from the input file."""

    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1)


class Parameter(BaseModel):
    """Describe a scalar argument and its optional allowed values."""

    model_config = ConfigDict(extra="forbid", strict=True)
    type: str
    description: str = ""
    enum: list | None = None

    def validate_value(self, value: Any) -> Any:
        """Check the type and allowed values, returning a typed value."""
        types = {"string": str, "number": float, "integer": int,
                 "boolean": bool, "null": type(None)}
        value = TypeAdapter(types[self.type]).validate_python(
            value, strict=True
        )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Numbers must be finite")
        if self.enum is not None and value not in self.enum:
            raise ValueError("Value is not in the allowed enum")
        return value

    @model_validator(mode="after")
    def check_schema(self) -> "Parameter":
        """Reject unsupported types and invalid enum definitions."""
        if self.type not in ("string", "number", "integer", "boolean", "null"):
            raise ValueError(f"Unsupported argument type: {self.type}")
        if self.enum is not None:
            if not self.enum:
                raise ValueError("An enum must contain at least one value")
            self.enum = [self.validate_value(value) for value in self.enum]
        return self


class FunctionDefinition(BaseModel):
    """Store a function with validated argument definitions."""

    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict
    returns: dict

    @model_validator(mode="after")
    def check_definition(self) -> "FunctionDefinition":
        """Validate names, parameter schemas, and return metadata."""
        for name in [self.name, *self.parameters]:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"Invalid name: {name!r}")
            if keyword.iskeyword(name):
                raise ValueError(f"A name cannot be a Python keyword: {name}")
        self.parameters = {
            name: Parameter.model_validate(schema)
            for name, schema in self.parameters.items()
        }
        if self.returns.get("type") not in (
            "string", "number", "integer", "boolean", "null", "array", "object"
        ):
            raise ValueError("Invalid return type")
        return self


class FunctionCallResult(BaseModel):
    """Store exactly the three fields required in the output."""

    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str
    name: str
    parameters: dict
