from sys import argv
from llm_sdk import Small_LLM_Model
from .parsing import Parsing
from .cdecod import ConstrainedDecoding
from .modules import FunctionsCallResult
from pathlib import Path
import time
import json


def transfer_list_to_file(
        result_to: list[FunctionsCallResult], path: str) -> None:
    """
    Write a list of FunctionsCallResult objects out to a JSON file.

    :param result_to: the list of validated results to write.
    :param path: the destination file path.
    """
    all_result = [x.model_dump() for x in result_to]

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        json.dump(all_result, file, indent=2)
        file.write("\n")


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


def main() -> None:
    """
    Entry point: parse CLI args, load and validate input files, run the
    constrained-decoding pipeline over every prompt, and write the results.
    """
    parsing = Parsing()
    parsing.parser(argv[1:])

    prompts = parsing.validation_v2(parsing.promts_file, 1)
    functions = parsing.validation_v2(
        parsing.functions_def_file, 2
    )

    model = Small_LLM_Model()
    decoder = ConstrainedDecoding(model, functions)

    results: list[FunctionsCallResult] = []
    for prompt_entry in prompts:
        try:
            result = decoder.run(prompt_entry.prompt)
            results.append(result)
        except ValueError as e:
            print(
                f"Warning: skipped prompt '{prompt_entry.prompt}' "
                f"due to an error: {e}"
            )

    transfer_list_to_file(results, parsing.output_file)
    print(f"Done. Wrote {len(results)} result(s) to {parsing.output_file}")


if __name__ == "__main__":
    try:
        start = time.perf_counter()
        main()
        end = time.perf_counter()
        print("Time taken:", end - start, "seconds")
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)
