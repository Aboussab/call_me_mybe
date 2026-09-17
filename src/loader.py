from __future__ import annotations
import sys
import json
from typing import Any
from pydantic import ValidationError
from .models import FunctionDefinition
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


def build_prompt(
    functions: list[FunctionDefinition],
    user_text: str,
) -> str:
    fn_descriptions = "\n".join(
        f"{fn.name}: {fn.description}\n"
        f"Parameters: {json.dumps({p: t.type for p, t in fn.parameters.items()})}"
        for fn in functions
    )

    return (
        f"You are a function calling assistant.\n\n"
        f"Available functions:\n"
        f"{fn_descriptions}\n\n"
        f"Example:\n"
        f"User request: What is the sum of 40 and 2?\n"
        f'Answer: {{"name": "fn_add_numbers", "parameters": {{"a": 40, "b": 2}}}}\n\n'
        f"If no available function matches the request, "
        f"the correct answer is fn_none\n"
        f"User request: What is the capital of Paris?\n"
        f'Answer: {{"name": "fn_none", "parameters": {{"none"}} }}\n\n'
        f"User request: {user_text}\n"
        f"Return ONLY the JSON function call.\n"
    )


def load_validated(path: str, schema: type[Any]) -> list[Any]:
    """Parse a JSON file and validate each entry against a Pydantic model."""
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(4)
    except json.JSONDecodeError:
        print(f"Invalid JSON: {path}")
        sys.exit(1)

    if not isinstance(raw, list):
        print(f"Expected a JSON list in {path}, got {type(raw).__name__}")
        sys.exit(1)

    try:
        return [schema(**entry) for entry in raw]
    except TypeError as exc:
        print(f"Invalid data in {path}\nError: {exc} ")
        sys.exit(1)
    except ValidationError as exc:
        print(f"Invalid data in {path}")
        print(f"Error: {exc.errors()[0]['msg']}")
        sys.exit(1)


class LLM:
    def __init__(self) -> None:
        try:
            self.model = Small_LLM_Model()
        except Exception as exc:
            raise RuntimeError(f"Failed to load LLM: {exc}")
