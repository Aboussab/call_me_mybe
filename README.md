*This project has been created as part of the 42 curriculum by aboussab*

## Description

`call_me_mybe` is a function-calling project built around constrained decoding with a local Qwen3-0.6B language model. Given a natural-language request and a JSON file describing the available functions, the program selects the most appropriate function and generates its parameters.

The main goal is to make the model produce a valid function call directly, instead of generating free-form text that must be repaired afterwards. The output is a JSON array containing one result per successfully processed prompt, with a function name and parameters matching the project schema.

The project is designed to run locally through the provided `llm_sdk` package. It uses the model's token encoding, logits, vocabulary and decoding primitives to enforce the allowed output at generation time.

## Instructions

### Requirements

- Python 3.10 or newer
- `uv`
- A local model setup supported by the provided `llm_sdk` package

### Installation

Install the project and its workspace dependency with `uv`:

```bash
uv sync
```

The root project and the local `llm_sdk` package are members of the same `uv` workspace. The dependency environment is therefore created and managed by `uv` rather than by a manually created virtual environment.

### Execution

Run the default example dataset with:

```bash
uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calling_results.json
```

The equivalent Make target is:

```bash
make run
```

The program loads the function definitions and prompts, processes every prompt, and writes the resulting JSON to `data/output/function_calling_results.json`. Input files must contain JSON lists matching their respective schemas.

### Quality checks

Run the configured lint and type checks with:

```bash
make lint
```

This runs `flake8` and `mypy` with strict-oriented options, including checks for untyped definitions and return types.

## Algorithm explanation

The algorithm performs constrained generation in several stages.

1. **Load and validate the schema.** Function definitions and user prompts are parsed into Pydantic models. A function definition contains its name, description, parameter names, parameter types and return type. Only functions present in `functions_definition.json` are eligible for selection.

2. **Build the function-selection prompt.** The available function descriptions and parameter types are included in a prompt together with the user's request. The model is asked to return only a JSON function call.

3. **Encode the current context.** The prompt is encoded with the SDK. The returned tensor contains a batch dimension, so the first batch is selected before passing token IDs to the constrained decoder.

4. **Mask logits during function-name generation.** At every generation step, the decoder obtains the model logits for the current token sequence. It masks every token that cannot continue one of the known function names. The next token is selected only from the remaining valid tokens. This prevents the model from spelling a function name that is absent from the schema and makes the function-selection phase deterministic with respect to the allowed vocabulary paths.

5. **Generate parameters one by one.** Once a function name has been selected, the decoder appends the JSON opening syntax and processes each declared parameter in schema order. The allowed tokens depend on the parameter type:

	 - **number:** numeric tokens are allowed, including the syntax needed for decimal values and a terminating JSON delimiter;
	 - **boolean:** only the valid boolean literals are allowed (`true` or `false`);
	 - **string:** tokens are generated between JSON quotes, with the closing quote and following delimiter constrained as part of the output;
	 - **integer:** numeric generation is used and the resulting value is converted to an integer before validation/output.

	 This type-specific generation is implemented through decoder classes for function names, numbers, strings and booleans. The orchestration layer chooses the decoder from the parameter schema and extends the token context after each value.

6. **Build and serialize the result.** The selected name, original prompt and generated parameters are assembled into a Pydantic `FunctionCallResult`. The collection of results is serialized as strict JSON. Invalid or failed generations are not written as fabricated successful calls.

## Design decisions

- **Pydantic for data models:** all input and output structures are represented by Pydantic models. This provides explicit schemas, readable validation errors and a single source of truth for prompts, parameter types, function definitions and function-call results.
- **Modular I/O and decoding:** loading/validation and prompt construction are separated from model interaction and constrained generation. In the current repository, `src/models.py` owns the Pydantic models, `src/loader.py` handles validated loading and prompt construction, `src/decoder_base.py` contains the constrained decoder implementations, `src/assembler.py` orchestrates one complete call, and `src/__main__.py` handles the command-line pipeline and output file. These modules correspond to the model, parsing, constrained-decoding and main responsibilities of the project.
- **Schema-driven behavior:** function names and parameter types are read from `functions_definition.json` rather than hard-coded in the generation loop. This keeps the decoder aligned with the input contract.
- **No `fn_none` fallback:** the intended output is strictly limited to the real functions declared in `functions_definition.json`. No fallback function named `fn_none` is added to the schema or used as a valid function definition. This avoids silently extending the contract and ensures that a successful result always refers to an actual declared function.
- **Local model through `llm_sdk`:** the project uses the supplied SDK instead of reimplementing model loading. The SDK exposes `encode`, `get_logits_from_input_ids`, `get_path_to_vocab_file` and `decode`, which are sufficient for the constrained generation loop.

## Performance analysis

Constrained decoding trades some raw generation speed for structural reliability. Each token requires a logits lookup and a validity mask, and parameter generation may require a separate constrained phase for every parameter. The implementation remains suitable for the project dataset: the target is to process the complete set of prompts in less than five minutes on the available 42 machine environment.

Precision is improved because invalid function names and invalid primitive values are rejected before they become part of the generated result. Pydantic validation adds a second boundary check after generation, while numeric coercion handles the distinction between generated numeric text and the declared `integer` or `number` type.

Reliability depends on the model, vocabulary and hardware being available to `llm_sdk`. The decoder is deliberately conservative: it prefers a valid schema-conforming token path over unconstrained sampling, and the output writer emits only successfully assembled results.

## Challenges faced

- **Limited disk quota on `/home`:** model and dependency caches can be large on a 42 machine. The practical solution is to redirect the caches to `/goinfre` before installing or running the project:

	```bash
	export UV_CACHE_DIR=/goinfre/$USER/uv-cache
	export HF_HOME=/goinfre/$USER/huggingface
	uv sync
	```

- **Vocabulary orientation:** the vocabulary file is provided as a `token_string -> token_id` mapping, while constrained decoding needs to test candidate token IDs and recover their token strings. The mapping must therefore be inverted once during initialization into `token_id -> token_string`.

- **Batched encoding output:** `encode()` returns token IDs with a batch dimension even when only one prompt is encoded. Selecting the first batch before converting it to a list avoids passing nested IDs into the logits and decoding functions.

- **Keeping generated text valid:** unconstrained next-token selection can produce malformed JSON or values with the wrong primitive type. Separate masks and decoder implementations for function names, numbers, booleans and strings solve this at the point of generation instead of relying only on post-processing.

- **Keeping the contract strict:** adding a synthetic fallback function would make the generated schema diverge from `functions_definition.json`. The implementation therefore keeps the set of selectable names tied to the definitions supplied by the user.

## Testing strategy

Validation was performed using the example prompts and function definitions shipped with the project. The checks include:

- running the five representative function-calling examples covering arithmetic, greeting, string manipulation and other typed parameters;
- checking that the generated file is valid JSON and that every function name exists in `functions_definition.json`;
- checking that every generated parameter has the declared name and compatible type (`number`, `boolean`, `string` or `integer`);
- validating input and output structures through the Pydantic models;
- running `flake8` and `mypy` in strict mode through `make lint`.

The complete sample input suite can be run with `make run`; its output is written to `data/output/function_calling_results.json` for inspection or automated comparison.

## Example usage

Run the project with the bundled files:

```bash
make run
```

Run it explicitly with custom paths:

```bash
uv run python -m src \
	--functions_definition path/to/functions_definition.json \
	--input path/to/prompts.json \
	--output path/to/results.json
```

Use the default command-line paths without specifying arguments:

```bash
uv run python -m src
```

Run the pipeline under the configured debugger entry point:

```bash
make debug
```

The resulting file has the following general shape:

```json
[
	{
		"prompt": "What is the sum of 2 and 3?",
		"name": "fn_add_numbers",
		"parameters": {
			"a": 2,
			"b": 3
		}
	}
]
```

## Resources

- [Qwen documentation](https://qwen.readthedocs.io/): background on the Qwen model family and local usage.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/): useful reference for tokenization, logits and autoregressive generation concepts.
- [Hugging Face constrained beam search documentation](https://huggingface.co/docs/transformers/en/generation_strategies): general background on restricting model generation to valid sequences.
- [JSON specification](https://www.json.org/json-en.html): reference for the syntax that the generated function calls must follow.
- [Pydantic documentation](https://docs.pydantic.dev/): reference for declarative Python validation and typed data models.
- [Python `json` documentation](https://docs.python.org/3/library/json.html): reference for parsing and serializing the input and output files.
- [uv documentation](https://docs.astral.sh/uv/): reference for Python dependency, environment and workspace management.

### AI used in the project

The project uses a local **Qwen3-0.6B** model through the provided `llm_sdk`. The model is used for the semantic parts of the task: understanding each natural-language request, choosing the matching function among the available definitions, and generating the value of each parameter. It is not trusted to define the output format by itself. The constrained decoder controls the allowed tokens, while Pydantic validates the resulting Python structures and the JSON writer produces the final file.

The model is therefore used in the function-selection and parameter-generation stages; `llm_sdk` provides model access and token-level operations; and the project code supplies the schema, masks, orchestration and validation that turn free-form language understanding into a strict function call.
