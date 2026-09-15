from sys import exit, argv
from .parsing import Parsing
from .modules import FunctionsCallResult
from llm_sdk import Small_LLM_Model
from .cdecod import ConstrainedDecoding
import time
import json
import sys
import traceback


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

    parse = Parsing()
    parse.parser(sys.argv)
    parse.validation_v1()
    model = Small_LLM_Model()

    constran = ConstrainedDecoding(model, parse.functions_def_file)

    user_prompt = "where is marthen lother king?"
    token_id = model.encode(f"You are a function-calling assistant. Given a user question and a list of available functions, choose the single function that best answers\
 the question.\nif there is none propper answer return a "fn_none"\n Available functions:\n{parse.functions_def_file}\n\n User question: {user_prompt}\n\nFunction name:")[0].tolist()
    start = time.perf_counter()
    print(constran.constrain_fct_name(token_id))
    end = time.perf_counter()
    print("Time taken:", end - start, "seconds")

if __name__ == "__main__":
    try:

        main()
        print("everythings is okey")
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
    except Exception as e:
        print(e)