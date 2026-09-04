import json
import sys


class Parsing:
    # def __init__(self, args):
    #    "docstring:"
    #    for i, arg in enumerate(args):
    #        if arg == "--input":
    #            self.promts_file = args[i + 1]
    #        elif arg == "--functions_definition":
    #            self.functions_def_file = args[i + 1]
    #        elif arg == "--output":
    #            self.output_file = args[i + 1]

    def loading(file):
        "docstring"
        try:
            with open(file) as f:
                data = json.load(f)
        except FileNotFoundError:
            print("there is an error while opening the fill")
        except json.JSONDecodeError:
            print("you should pass a file with the json format")
        if isinstance(data, list) and all(isinstance(n, dict) for n in data):
            print("pleas make sure the data passed is correct")
        return data
