from sys import exit, argv
from .parsing import Parsing
from .modules import FunctionsCallResult
import json


def write_results(
     result_to: list[FunctionsCallResult], path: str) -> None:
    all_result = []
    for x in result_to:
        all_result.append(x.model_dump())
    try:
        with open(path, "w") as file:
            json.dump(all_result, file, indent=2)
            file.write("\n")
    except FileNotFoundError as e:
        print(e)
        exit(1)


def main():
    from llm_sdk import Small_LLM_Model

    model = Small_LLM_Model()
    tokens = model.encode("What is the sum of 2 and 3?")
    print(type(tokens))
    print(tokens)


if __name__ == "__main__":
    try:
        #arg = Parsing()
        #arg.parser(argv)
        #arg.validation_v1()
        main()
        print("everythings is okey")
    except Exception as e:
        print(e)
        exit(1)
