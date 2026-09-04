from pydantic import BaseModel, Field
from typing import Literal


class Promt_par(BaseModel):
    "docs"
    prompt: str = Field(min_length=1)


class Parametre_type(BaseModel):
    type: Literal["string", "boolean", "number"]


class Func_def(BaseModel):
    "docstring"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str = Field(min_length=1)
    parameters: dict[str, Parametre_type]
    returns: dict[str, str]


class Fct_call_result(BaseModel):
    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, str]
