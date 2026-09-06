from .modules import FunctionsDefinition, FunctionsCallResult
from .modules import Prompt
from typing import Any
from pathlib import Path
from pydantic import ValidationError
from sys import exit
import json


class Parsing():

    def __init__(self):
        """
        lool
        """
        self.promts_file: str | None = None
        self.functions_def_file: str | None = None
        self.output_file: str | None = None

    def parser(self, args: list) -> None:
        """
        parser is a fct that take the argument passed to the programe
        and check if every
        """
        try:
            for i, arg in enumerate(args):
                if arg == "--input":
                    self.promts_file = args[i + 1]
                elif arg == "--functions_definition":
                    self.functions_def_file = args[i + 1]
                elif arg == "--output":
                    self.output_file = args[i + 1]
        except IndexError:
            print("pleas make sure to pass the right argument with\
the following flags:\
\n  [--functions_definition <function_definition_file>] [--input <input_file>]\
[--output <output_file>]\
\n       --functions_definition data/input/functions_definition.json\
\n       --input data/input/function_calling_tests.json\
\n       --output data/output/function_calls.json")
            exit(1)
        if (
             not self.promts_file or not self.functions_def_file
             or not self.output_file
             ):
            print("pleas make sure to pass the right argument with\
the following flags:\
\n  [--functions_definition <function_definition_file>] [--input <input_file>]\
[--output <output_file>]\
\n       --functions_definition data/input/functions_definition.json\
\n       --input data/input/function_calling_tests.json\
\n       --output data/output/function_calls.json")
            exit(1)

    @staticmethod
    def loading(file: str) -> list[Any]:
        """
        loading is a function that take the file path to be loeaand return it
        as list of dict.
        """
        try:
            with open(file) as f:
                data = json.load(f)
        except FileNotFoundError:
            print("ERROR DETCTED 'FileNotFoundError': make sure u passed the right path")
            exit(1)
        except json.JSONDecodeError:
            print("ERROR DETCTED: you should pass a file with the json format")
            exit(1)
        if not (
             isinstance(data, list) and all(isinstance(n, dict) for n in data)
             ):
            print("ERROR DETCTED: Pleas make sure the data passed is correct")
            exit(1)
        return data

    def validation_v2(self, path: str, n: int) -> list:
        """
        lool
        """
        file = self.loading(path)
        if (n == 1):
            file_as_list: list[Prompt] = []
            for entry in file:
                try:
                    parsed = Prompt(**entry)
                    file_as_list.append(parsed)
                except ValidationError:
                    print(f"ERROR DETCTED (FILE FORMAT): Pleas make\
 sure that the data in your  \"{Path(path).name}\"is on the correct format.")
                    exit(1)
        elif (n == 2):
            file_as_list: list[FunctionsDefinition] = list()
            seen_names: set[str] = set()
            for entry in file:
                try:
                    parsed = FunctionsDefinition(**entry)
                    if parsed.name in seen_names:
                        raise ValueError(
                             f"Duplicate function name found: '{parsed.name}'")
                    seen_names.add(parsed.name)
                    file_as_list.append(parsed)
                except ValidationError:
                    print(f"ERROR DETCTED (FILE FORMAT): Pleas make\
 sure that the data in your  \"{Path(path).name}\"is on the correct format.")
                    exit(1)
                except ValueError as e:
                    print(f"ERROR DETCTED: {e}")
        else:
            raise ValueError("Invalid validation type")
        return (file_as_list)

    def validation_v1(self) -> None:
        self.validation_v2(self.promts_file, 1)
        self.validation_v2(self.functions_def_file, 2)
