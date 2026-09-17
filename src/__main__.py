from __future__ import annotations
import argparse
import json
import os
import time

from pydantic import ValidationError

from .assembler import Assembler
from .loader import load_validated, LLM
from .models import FunctionDefinition, Prompt


def run_pipeline(functions_path: str, input_path: str, output_path: str) -> None:
    fn_defs = load_validated(functions_path, FunctionDefinition)
    prompts = load_validated(input_path, Prompt)
    llm = LLM()

    with open(llm.model.get_path_to_vocab_file()) as fh:
        vocabulary = json.load(fh)

    collected: list[dict] = []

    for entry in prompts:
        t0 = time.perf_counter()

        try:
            outcome = Assembler.run_one(llm, vocabulary, fn_defs, entry.prompt)
        except Exception as err:
            print(f"Failed on prompt {entry.prompt!r}: {err}")
            outcome = None

        print(outcome)
        print(f"Took {time.perf_counter() - t0:.2f} seconds")

        if outcome is not None:
            collected.append(outcome.model_dump())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(collected, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    args = parser.parse_args()
    run_pipeline(args.functions_definition, args.input, args.output)


if __name__ == "__main__":
    wall_start = time.perf_counter()
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except ValidationError as exc:
        print(f"Error: {exc.errors()[0]['msg']}")
    except Exception as exc:
        print(f"Error:\n  {type(exc).__name__}: {exc}")
    finally:
        total_time = (time.perf_counter() - wall_start) / 60
        print(f"TOTAL DURATION: {total_time:.2f} mins")
