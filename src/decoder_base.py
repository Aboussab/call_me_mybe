from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from numpy.typing import NDArray
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class ConstraintDecoder(ABC):
    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]) -> None:
        self.model = model
        self.vocab = vocab

    def apply_mask(
        self,
        logits: NDArray[np.float32],
        permitted: set[int] | list[int],
    ) -> NDArray[np.float32]:
        masked = np.full_like(logits, -np.inf)
        for token_id in permitted:
            masked[token_id] = logits[token_id]
        return masked

    @abstractmethod
    def extract(self, ids: list[int]) -> Any:
        pass


class FunctionNameDecoder(ConstraintDecoder):
    def __init__(
        self,
        model: Small_LLM_Model,
        vocab: dict[str, int],
        fn_defs: list,
    ) -> None:
        super().__init__(model, vocab)
        self.fn_names = [fn.name for fn in fn_defs] + ["fn_none"]
        self.fn_token_seqs = [
            model.encode(name)[0].tolist() for name in self.fn_names
        ]

    def extract(self, ids: list[int]) -> Any:
        generated: list[int] = []
        remaining = list(self.fn_token_seqs)

        while True:
            cursor = len(generated)
            valid_next = {
                seq[cursor]
                for seq in remaining
                if cursor < len(seq)
            }

            if not valid_next:
                return None

            logits = self.model.get_logits_from_input_ids(ids + generated)
            chosen = int(np.argmax(self.apply_mask(logits, valid_next)))
            generated.append(chosen)

            remaining = [
                seq for seq in remaining
                if seq[:cursor + 1] == generated
            ]

            if len(remaining) == 1 and remaining[0] == generated:
                ids.extend(generated)
                match_idx = self.fn_token_seqs.index(generated)
                return str(self.fn_names[match_idx])


class NumberDecoder(ConstraintDecoder):
    DIGIT_CHARS = set("0123456789")
    TOKEN_LIMIT = 20

    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]) -> None:
        super().__init__(model, vocab)
        self.digit_ids   = [tid for ch, tid in vocab.items() if ch in self.DIGIT_CHARS]
        self.dot_ids     = [tid for ch, tid in vocab.items() if ch == "."]
        self.minus_ids   = [tid for ch, tid in vocab.items() if ch == "-"]
        self.comma_ids   = [tid for ch, tid in vocab.items() if ch == ","]
        self.rbrace_ids  = [tid for ch, tid in vocab.items() if ch == "}"]

    def extract(self, ids: list[int]) -> Any:
        accumulated = ""

        for _ in range(self.TOKEN_LIMIT):
            valid = list(self.digit_ids)

            if not accumulated:
                valid += self.minus_ids
            if "." not in accumulated:
                valid += self.dot_ids
            if accumulated and any(ch in accumulated for ch in self.DIGIT_CHARS):
                valid += self.comma_ids
                valid += self.rbrace_ids

            logits = self.model.get_logits_from_input_ids(ids)
            chosen = int(np.argmax(self.apply_mask(logits, valid)))
            decoded_tok = self.model.decode([chosen])

            if "," in decoded_tok or "}" in decoded_tok:
                stop_char = "," if "," in decoded_tok else "}"
                before_stop = decoded_tok[:decoded_tok.index(stop_char)]
                if before_stop:
                    ids.extend(self.model.encode(before_stop)[0].tolist())
                return accumulated + before_stop

            accumulated += decoded_tok
            ids.append(chosen)

        return accumulated


class StringDecoder(ConstraintDecoder):
    TOKEN_LIMIT = 40
    REPLACEMENTS = {"asterisk": "*", "asterisks": "*"}

    def extract(self, ids: list[int]) -> Any:
        non_newline_ids = [tid for ch, tid in self.vocab.items() if ch != "\n"]
        text = ""

        for _ in range(self.TOKEN_LIMIT):
            logits = self.model.get_logits_from_input_ids(ids)
            chosen = int(np.argmax(self.apply_mask(logits, non_newline_ids)))
            decoded_tok = self.model.decode([chosen])

            if '"' in decoded_tok:
                ids.append(chosen)
                text += decoded_tok[:decoded_tok.index('"')]
                return self.REPLACEMENTS.get(text.strip().lower(), text)

            text += decoded_tok
            ids.append(chosen)

        return self.REPLACEMENTS.get(text.strip().lower(), text)


class BoolDecoder(ConstraintDecoder):
    BOOL_STRINGS = ["true", "false"]

    def __init__(self, model: Small_LLM_Model, vocab: dict[str, int]) -> None:
        super().__init__(model, vocab)
        self.bool_token_seqs = [
            model.encode(val)[0].tolist() for val in self.BOOL_STRINGS
        ]

    def extract(self, ids: list[int]) -> bool:
        lead_tokens = [seq[0] for seq in self.bool_token_seqs]
        logits = self.model.get_logits_from_input_ids(ids)
        chosen = int(np.argmax(self.apply_mask(logits, lead_tokens)))

        matched = lead_tokens.index(chosen)
        ids.extend(self.bool_token_seqs[matched])
        return self.BOOL_STRINGS[matched] == "true"
