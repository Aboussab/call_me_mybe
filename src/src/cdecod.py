from abc import ABC
import json
from llm_sdk import Small_LLM_Model
from .modules import FunctionsDefinition


class ConstrainedDecoding():
    def __init__(self, model: Small_LLM_Model,
                 functions_def: list[FunctionsDefinition]
                 ):
        self.model = model
        self.functions_def = functions_def
        self.id_to_token = self._load_vocab()

    def fct_bach_ghan9ado_jsonformat():
        """this fct porpuse is to guid the output to print a json format"""
        pass

    def building_prompt(self, user_prompt):
        """
        this fct porpuse is to give the llm a specfique prompt where i put it
        in the context about what i need

        :param self: here i pass self to have this methode as instance method.
        :param user_prompt: this is the specifique methode a user sent to me
            to answer on it.
        """

        return (
            "You are a function-calling assistant. Given a user question and"
            " a list of available functions, choose the single function"
            " that best answers the question.\n\n"
            f"Available functions:\n{self.functions_def}\n\n"
            f"User question: {user_prompt}\n\n"
            "Function name:"
            )

# -----------------from here we have the fct_constraining.---------------------
    def constrain_fct_name(self, prompt_token_ids: list[int]) -> str:
        """
        Generate a function name token-by-token, constrained so the output
        can only ever match one of the known function names.
        """

        fn_listed = [
            fn.name for fn in self.functions_def
            ] + ["fn_none"]

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

    def build_parameter_prompt(
            self,
            user_question: str,
            function_name: str,
            param_name: str,
            param_type: str,
            ) -> str:
        """Build the prompt text used to generate one parameter's value."""
        return (
            f"Question: {user_question}\n"
            f"Function: {function_name}\n"
            f"Parameter \"{param_name}\" (type: {param_type}):"
        )
    
    def generate_parameter_value(self, context_ids, param_type):
        pass

    def _generate_number(self, user_promt: list):
        allowed_chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]
        generated_answer = ""
        current_token_ids = list(user_promt)
        i = 0
        max_token = 40
        for i in range(max_token):
            logits = self.model.get_logits_from_input_ids(current_token_ids)

            best_token_id = None
            best_score = float("-inf")
            for token_id, raw_token in self.id_to_token.items():
                token_text = self._clean_token_text(raw_token)
                if (not all(char in allowed_chars for char in token_text)):
                    continue
                if token_text == "":
                    continue

                score = logits[token_id]
                if score > best_score:
                    best_score = score
                    best_token_id = token_id

            if best_token_id is None:
                break

            generated_answer += self._clean_token_text(self.id_to_token[best_token_id])
            current_token_ids.append(best_token_id)

        if generated_answer == "":
            raise ValueError("Could not generate a valid number: got empty text")

        try:
            return float(generated_answer)
        except ValueError:
            raise ValueError(
                f"Generated text is not a valid number: '{generated_answer}'"
            )

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
                token_text = self._clean_token_text(raw_token) # why did we check this and knowing that changing a token one char will change it id 
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

#     def _generate_string():
#         pass
