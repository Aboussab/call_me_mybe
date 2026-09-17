from __future__ import annotations
from .models import FunctionCallResult, FunctionDefinition
from .loader import build_prompt, LLM
from .decoder_base import (
    ConstraintDecoder,
    FunctionNameDecoder,
    NumberDecoder,
    StringDecoder,
    BoolDecoder,
)


def _pick_decoder(
    param_type: str,
    model,
    vocab: dict[str, int],
) -> ConstraintDecoder:
    if param_type in ("number", "integer"):
        return NumberDecoder(model, vocab)
    if param_type == "boolean":
        return BoolDecoder(model, vocab)
    return StringDecoder(model, vocab)


def _coerce_value(raw_value, param_type: str):
    if param_type == "integer":
        return int(float(raw_value))
    if param_type == "number":
        return float(raw_value)
    return raw_value


class Assembler:

    @staticmethod
    def run_one(
        llm: LLM,
        vocab: dict[str, int],
        functions: list[FunctionDefinition],
        prompt_text: str,
    ) -> FunctionCallResult | None:
        full_prompt = build_prompt(functions, prompt_text)
        token_ids = llm.model.encode(full_prompt)[0].tolist()

        name_dec = FunctionNameDecoder(llm.model, vocab, functions)
        fn_name = name_dec.extract(token_ids)

        if fn_name is None:
            return None

        if fn_name == "fn_none":
            return FunctionCallResult(
                prompt=prompt_text,
                name="fn_none",
                parameters={},
            )

        matched_fn = next(f for f in functions if f.name == fn_name)
        resolved_params: dict = {}

        for idx, (param_name, param_schema) in enumerate(matched_fn.parameters.items()):
            opening = ("{" if idx == 0 else ", ") + f'"{param_name}": '
            if param_schema.type == "string":
                opening += '"'
            token_ids.extend(llm.model.encode(opening)[0].tolist())

            dec = _pick_decoder(param_schema.type, llm.model, vocab)
            raw = dec.extract(token_ids)
            resolved_params[param_name] = _coerce_value(raw, param_schema.type)

        return FunctionCallResult(
            prompt=prompt_text,
            name=fn_name,
            parameters=resolved_params,
        )
