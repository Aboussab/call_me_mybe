from modules import FunctionsDefinition
from sys import exit
import json


def loading(file):
    "docstring"
    try:
        with open(file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print("there is an error while opening the fill")
        exit(0)
    except json.JSONDecodeError:
        print("you should pass a file with the json format")
        exit(0)
    if isinstance(data, list) and all(isinstance(n, dict) for n in data):
        print("pleas make sure the data passed is correct")
    return data


def main():
    file = loading("./functions_definition.json")

    for entry in file:
        parsed = FunctionsDefinition(**entry)
        print(parsed.name, "→ OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
