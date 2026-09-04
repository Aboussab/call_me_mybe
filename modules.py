from pydantic import BaseModel, Field


class func_def(BaseModel):
    "docstring"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9]*$")
    description: str = Field(min_length=1)
    parameters: dict[str, str]
    returns: dict[str, str]


class Promt_par(BaseModel):
    "docs"
    prompte: str = Field(min_length=1)


class fct_call_result():
    prompte: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, str]
