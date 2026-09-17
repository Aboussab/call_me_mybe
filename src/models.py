from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class Prompt(BaseModel):
    """Holds a single user-facing prompt string."""
    prompt: str = Field(min_length=1)


class ParamSchema(BaseModel):
    """Describes the expected type of a single function parameter."""
    type: Literal["string", "integer", "number", "boolean"]


class FunctionDefinition(BaseModel):
    """Full specification of a callable function including its parameters."""
    name: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$"
    )
    description: str = Field(min_length=1)
    parameters: dict[str, ParamSchema]
    returns: ParamSchema


class FunctionCallResult(BaseModel):
    """Represents the outcome of resolving a prompt to a function call."""
    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, Any]

    @field_validator("parameters")
    @classmethod
    def cap_large_numbers(cls, v: dict[str, Any]) -> dict[str, Any]:
        ceiling = 10**9 - 1
        return {
            key: min(val, ceiling) if isinstance(val, (int, float)) else val
            for key, val in v.items()
        }
