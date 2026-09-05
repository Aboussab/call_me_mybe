from modules import FunctionsDefinition
from typing import Any
from pydantic import ValidationError
from sys import exit
import json


class Parsing():

    def __init__(self):
        self.promts_fi = None
        self.functions_def_file = None
        self.output_file = None

    def parser(self, args):
        """
        parser is a fct that take the argument passed to the programe
        and check if every  
        """
        for i, arg in enumerate(args):
            if arg == "--input":
                self.promts_fi = args[i + 1]
            elif arg == "--functions_definition":
                self.functions_def_file = args[i + 1]
            elif arg == "--output":
                self.output_file = args[i + 1]

    def loading(file: str) -> list[Any]:
        """
        loading is a function that take the file path to be loeaand return it
        as list of dict.
        """
        try:
            with open(file) as f:
                data = json.load(f)
        except FileNotFoundError:
            print("ERROR DETCTED: there is an error while opening the fill")
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

    def validation(self, path: str) -> list:
        file = self.loading(path)

        file_as_list = []
        for entry in file:
            try:
                parsed = FunctionsDefinition(**entry)
                file_as_list.append(parsed)
            except ValidationError:
                print("ERROR DETCTED: Pleas make sure that the data in\
your json file is on the correct format.")
                exit(1)
        return