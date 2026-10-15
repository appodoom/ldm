"""HF tokenizer with direct, exact atom-ID mapping."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import PreTrainedTokenizer

from .derbake_ldm import HITS  # HF caches only direct relative imports.
from .vocabulary import AtomicVocabulary, vocabulary_hash


class LDMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str | None = None, tokens: list[str] | None = None, **kwargs):
        if vocab_file:
            payload = json.loads(Path(vocab_file).read_text(encoding="utf-8"))
            tokens = payload["tokens"]
            if vocabulary_hash(tokens) != payload["sha256"]:
                raise ValueError("Vocabulary hash mismatch")
        if tokens is None:
            raise ValueError("Pass vocab_file or tokens")
        self.atomic = AtomicVocabulary(tokens)
        kwargs.setdefault("pad_token", "<PAD>")
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.atomic.tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.atomic.to_id)

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        atoms = text.split()
        self.atomic.encode(atoms)
        return atoms

    def _convert_token_to_id(self, token: str) -> int:
        return self.atomic.encode([token])[0]

    def _convert_id_to_token(self, index: int) -> str:
        return self.atomic.decode([index])[0]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            raise ValueError("LDM does not accept paired sequences")
        return list(token_ids_0)

    def encode_atoms(self, atoms: list[str]) -> list[int]:
        return self.atomic.encode(atoms)

    def decode_atoms(self, ids: list[int]) -> list[str]:
        return self.atomic.decode(ids)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        path = Path(save_directory) / (f"{filename_prefix}-vocab.json" if filename_prefix else "vocab.json")
        path.write_text(json.dumps({"tokens": list(self.atomic.tokens), "sha256": self.atomic.hash},
                                   indent=2) + "\n", encoding="utf-8")
        return (str(path),)
