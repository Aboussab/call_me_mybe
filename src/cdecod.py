import json
from llm_sdk import Small_LLM_Model
from .modules import FunctionsDefinition, FunctionsCallResult


class ConstrainedDecoding():
    def __init__(self, model: Small_LLM_Model,
                 functions_def: list[FunctionsDefinition]
                 ):
        self.model = model
        self.functions_def = functions_def
        self.id_to_token = self._load_vocab()

    def run(self, prompt: str) -> FunctionsCallResult:
        """
        Run the full constrained-decoding pipeline for one user prompt:
        select the matching function, generate a value for each of its
        parameters, and return the assembled result.

        :param self: instance method.
        :param prompt: the original natural-language user prompt.
        :return: a validated FunctionsCallResult ready to be written out.
        """
        # step 1: select which function to call
        selection_prompt = self.building_prompt(prompt)
        selection_tokens = self.model.encode(selection_prompt)
        selection_token_ids = selection_tokens[0].tolist()
        function_name = self.constrain_fct_name(selection_token_ids)

        # step 2: look up the full schema for that function
        selected_function = self._find_function_by_name(function_name)

        # step 3: generate a value for each parameter, one at a time
        parameters: dict[str, str | float | bool] = {}
        for param_name, param_schema in selected_function.parameters.items():
            param_prompt = self.build_parameter_prompt(
                user_question=prompt,
                function_name=function_name,
                param_name=param_name,
                param_type=param_schema.type,
                filled_params=parameters,
                param_description=getattr(param_schema, "description", None),
            )
            param_tokens = self.model.encode(param_prompt)
            param_token_ids = param_tokens[0].tolist()

            if param_schema.type == "number":
                value = self._generate_number(param_token_ids)
            elif param_schema.type == "boolean":
                value = self._generate_boolean(param_token_ids)
            elif param_schema.type == "string":
                value = self._generate_string(param_token_ids)
            else:
                raise ValueError(
                    f"Unsupported parameter type '{param_schema.type}' "
                    f"for parameter '{param_name}'"
                )

            parameters[param_name] = value

        # step 4: assemble and validate the final result
        return FunctionsCallResult(
            prompt=prompt,
            name=function_name,
            parameters=parameters,
        )

    def building_prompt(self, user_prompt: str) -> str:
        """
        Build the prompt used to select the single best-matching function
        for a user question.
        """

        fn_lines = "\n".join(
            f"- {fn.name}({', '.join(p for p in fn.parameters)}): "
            f"{fn.description}"
            for fn in self.functions_def
        )
        return (
            "You are a function-calling assistant. Your ONLY job is to pick "
            "the single function name that best answers the user's question.\n"
            "Answer with the function name ONLY — no punctuation, no "
            "explanation, no extra words.\n\n"
            "Available functions:\n"
            f"{fn_lines}\n\n"
            "Examples:\n"
            "Question: What is the sum of 4 and 9?\n"
            "Answer: fn_add_numbers\n\n"
            "Question: Say hello to Alice\n"
            "Answer: fn_greet\n\n"
            f"Question: {user_prompt}\n"
            "Answer:"
        )

    def build_parameter_prompt(
            self,
            user_question: str,
            function_name: str,
            param_name: str,
            param_type: str,
            filled_params: dict | None = None,
            param_description: str | None = None,
            ) -> str:
        """
        Build the prompt used to generate the value of one specific
        parameter for the already-selected function.
        :param filled_params: parameters of this same function already
            resolved in previous calls (name -> value).
        :param param_description: optional human-readable description of
            this parameter, if the schema provides one — critical for
            semantically ambiguous parameters like "regex" or "replacement".
        """
        filled = filled_params or {}
        filled_lines = (
            "\n".join(f'- "{k}" = {v}' for k, v in filled.items())
            if filled else "(none yet)")
        description_line = (
            f'Parameter meaning: {param_description}\n' if param_description else ""
        )

        base = (
            "You are a function-calling assistant. A function has already "
            "been selected. Extract or derive ONE parameter's value from "
            "the user's question below. The value must be grounded in the "
            "question text — never copy a number or word from these "
            "instructions or from any example.\n\n"
            f"User question: {user_question}\n"
            f"Selected function: {function_name}\n"
            f"{description_line}"
            f"Parameters already filled for this function:\n{filled_lines}\n\n"
            f"Now provide the value for parameter \"{param_name}\" "
            f"(type: {param_type}).\n\n"
        )

        if param_type == "string":
            return base + (
                "Answer with the value wrapped in double quotes, "
                "no explanation.\n\n"
                f"Value for \"{param_name}\": \""
            )

        if param_type == "number":
            return base + (
                "Answer with the number copied exactly from the question "
                "above, as digits only — no words, no quotes, no units.\n\n"
                f"Value for \"{param_name}\":"
            )

        return base + (
            "Answer with the raw value ONLY — no quotes, no explanation.\n\n"
            f"Value for \"{param_name}\":"
        )

    def _find_function_by_name(self, name: str) -> FunctionsDefinition:
        """
        Look up the full FunctionsDefinition object matching the given
        function name, out of the ones this decoder was built with. 
        :param self: instance method.
        :param name: the function name to look up (e.g. "fn_add_numbers"),
            expected to be one of the names in self.functions_def.
        :return: the matching FunctionsDefinition object.
        :raises ValueError: if no function with this name exists in
            self.functions_def.
        """ 
        for fn in self.functions_def:
            if fn.name == name:
                return fn
        raise ValueError(
            f"Selected function '{name}' was not found in the known "
            f"function definitions."
        )

# -----------------from here we have the fct_constraining.---------------------
    def constrain_fct_name(self, prompt_token_ids: list[int]) -> str:
        """
        Generate a function name token-by-token, constrained so the output
        can only ever match one of the known function names.
        """

        fn_listed = [
            fn.name for fn in self.functions_def
            ]

        generated_name = ""
        current_token_ids = list(prompt_token_ids)

        max_steps = 50
        for _ in range(max_steps):

            if generated_name in fn_listed:
                return generated_name
            logits = self.model.get_logits_from_input_ids(current_token_ids)

            best_token_id = None
            best_score = float("-inf")

            for token_id, raw_token in self.id_to_token.items():
                token_text = self._clean_token_text(raw_token)
                maybe_fctname = generated_name + token_text

                it_fit = any(
                    name.startswith(maybe_fctname) for name in fn_listed
                )
                if not it_fit:
                    continue

                score = logits[token_id]
                if score > best_score:
                    best_score = score
                    best_token_id = token_id

            if best_token_id is None:
                raise ValueError(
                    f"Could not continue generating a valid function name "
                    f"from partial text: '{generated_name}'"
                )

            generated_name += self._clean_token_text(
                self.id_to_token[best_token_id])
            current_token_ids.append(best_token_id)

        raise ValueError(
            f"Failed to generate a complete function name within \
{max_steps} steps, "
            f"got partial text: '{generated_name}'"
        )

    def _clean_token_text(self, txt: str):
        """here we have an str encoded and we should replace the charachter Ġ
        is replaced with its real meaning"""
        return txt.replace("Ġ", " ")

    def _load_vocab(self) -> dict[int, str]:
        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        return {v: k for k, v in token_to_id.items()}

# -----------------from heere we have the parameter constrain:-----------------

    def _generate_number(self, user_prompt: list) -> float:
        """Generate a JSON-number-shaped value, token by token, using constrained decoding."""
        generated_answer = ""
        current_token_ids = list(user_prompt)

        max_steps = 20
        for _ in range(max_steps):
            logits = self.model.get_logits_from_input_ids(current_token_ids)

            # unconstrained top choice — tells us what the model actually
            # wants to do next, digit or not
            global_best_id = max(range(len(logits)), key=lambda i: logits[i])

            best_token_id = None
            best_score = float("-inf")

            for token_id, raw_token in self.id_to_token.items():
                token_text = self._clean_token_text(raw_token)

                if token_text == "":
                    continue
                if not self._check_if_its_a_valid_number(generated_answer, token_text):
                    continue

                score = logits[token_id]
                if score > best_score:
                    best_score = score
                    best_token_id = token_id

            has_digit = any(c.isdigit() for c in generated_answer)
            if has_digit and global_best_id != best_token_id:
                global_best_text = self._clean_token_text(self.id_to_token[global_best_id])
                if not self._check_if_its_a_valid_number(generated_answer, global_best_text):
                    break  # model wants to stop and we have a usable number

            if best_token_id is None:
                break

            generated_answer += self._clean_token_text(self.id_to_token[best_token_id])
            current_token_ids.append(best_token_id)

        if generated_answer == "":
            raise ValueError("Could not generate a valid number: got empty text")

        try:
            return float(generated_answer)
        except ValueError:
            raise ValueError(f"Generated text is not a valid number: '{generated_answer}'")

    def _check_if_its_a_valid_number(self, generated_so_far: str, token_text: str) -> bool:
        """Check if appending token_text to generated_so_far keeps it a valid,
        still-incomplete JSON number."""
        allowed_chars = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"}

        for char in token_text:
            if char not in allowed_chars:
                return False

        candidate = generated_so_far + token_text

        # '-' allowed only as the very first character
        minus_count = candidate.count("-")
        if minus_count > 1:
            return False
        if minus_count == 1 and not candidate.startswith("-"):
            return False

        dot_count = candidate.count(".")
        if dot_count > 1:
            return False
        if dot_count == 1 and (candidate.startswith(".") or candidate.startswith("-.")):
            return False

        return True

    def _generate_boolean(self, prompt_token_id: list) -> bool:
        """
        Docstring for _generate_boolean
        """

        answer_list = ["true", "false"]
        generated_answer = ""
        current_token_ids = list(prompt_token_id)

        max_steps = 15
        for _ in range(max_steps):

            if generated_answer in answer_list:
                return generated_answer == "true"
            logits = self.model.get_logits_from_input_ids(current_token_ids)

            best_token_id = None
            best_score = float("-inf")

            for token_id, raw_token in self.id_to_token.items():
                token_text = self._clean_token_text(raw_token)
                argument = generated_answer + token_text

                it_fit = any(
                    name.startswith(argument) for name in answer_list
                )
                if not it_fit:
                    continue

                score = logits[token_id]
                if score > best_score:
                    best_score = score
                    best_token_id = token_id
            if best_token_id is None:
                raise ValueError(
                    f"Could not continue generating a valid argument"
                    f"from partial text: '{generated_answer}'"
                )
            generated_answer += self._clean_token_text(
                self.id_to_token[best_token_id])
            current_token_ids.append(best_token_id)
        raise ValueError(
            f"Failed to generate a complete function name within \
{max_steps} steps, "
            f"got partial text: '{generated_answer}'"
        )

    def _generate_string(self, user_prompt: list) -> str:
        """Generate a free-form string value, stopping once a closing quote appears."""
        generated_answer = ""
        current_token_ids = list(user_prompt)

        max_steps = 40
        for _ in range(max_steps):
            logits = self.model.get_logits_from_input_ids(current_token_ids)

            best_token_id = None
            best_score = float("-inf")

            for token_id, raw_token in self.id_to_token.items():
                token_text = self._clean_token_text(raw_token)

                if token_text == "":
                    continue

                score = logits[token_id]
                if score > best_score:
                    best_score = score
                    best_token_id = token_id

            if best_token_id is None:
                raise ValueError(
                    f"Could not continue generating a string from partial "
                    f"text: '{generated_answer}'"
                )

            token_text = self._clean_token_text(self.id_to_token[best_token_id])

            if '"' in token_text:
                generated_answer += token_text.split('"')[0]
                return generated_answer

            generated_answer += token_text
            current_token_ids.append(best_token_id)

        return generated_answer
