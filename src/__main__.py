from modules import FunctionsDefinition
from typing import Any
from pydantic import ValidationError
from sys import exit
import json


def loading(file: str) -> list[Any]:
    """
    loading is a function that take the file path to be loeaand return it
    as list of dict.
    """
    try:
        with open(file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print("there is an error while opening the fill")
        exit(1)
    except json.JSONDecodeError:
        print("you should pass a file with the json format")
        exit(1)
    if not (isinstance(data, list) and all(isinstance(n, dict) for n in data)):
        print("ERROR DETCTED: Pleas make sure the data passed is correct")
        exit(1)
    return data


def validation():
    file = loading("./data/input/functions_definition.json")

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


if __name__ == "__main__":
    try:
        validation()
    except Exception as e:
        print(e)
        exit(1)
