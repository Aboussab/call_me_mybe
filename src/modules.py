from pydantic import BaseModel, Field
from typing import Literal, Any


class Prompt(BaseModel):
    """
    Promt is a class where tarnsfer every prompt file content into a python
    object.
    """

    prompt: str = Field(min_length=1)


class ParametreType(BaseModel):
    """
    ParametreType is a class where i specified every paramater type can be
    passedinside the FunctionsDefinition file.
    """

    type: Literal["string", "boolean", "number", "integer"]


class FunctionsDefinition(BaseModel):
    """
    FunctionsDefinition is a class where we tarnsfer every
    functions_definition file content into a python object.
    """

    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str = Field(min_length=1)
    parameters: dict[str, ParametreType]
    returns: ParametreType


class FunctionsCallResult(BaseModel):
    """
    FunctionsCallResult is a python object that hold the result of the LLM
    and the porpose is to tarnsfer it into a json file after.
    """

    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: dict[str, Any]
