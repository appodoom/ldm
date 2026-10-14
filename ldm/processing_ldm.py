"""Shared .derbake codec + selected rhythm grammar + atomic tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, ProcessorMixin
from transformers.utils.hub import cached_file

from .derbake_ldm import RhythmSequence, parse_derbake, write_derbake
from .schema_ldm import RhythmSchema
from .tokenization_ldm import LDMTokenizer


class LDMDerbakeProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizer"

    def __init__(self, tokenizer: LDMTokenizer, dataset_variant: str = "with_subd",
                 max_beats_per_cycle: int = 16):
        super().__init__(tokenizer)
        self.dataset_variant = dataset_variant
        self.max_beats_per_cycle = int(max_beats_per_cycle)
        self.schema = RhythmSchema(dataset_variant, self.max_beats_per_cycle)

    def __call__(self, *, derbake_path: str | Path, return_tensors: str | None = None):
        document = parse_derbake(derbake_path, max_beats_per_cycle=self.max_beats_per_cycle)
        atoms = self.schema.encode(document.rhythm)
        ids = self.tokenizer.encode_atoms(atoms)
        if return_tensors is None:
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")
        return {"input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}

    def decode_generation(self, continuation_ids, *, expected_num_cycles: int) -> RhythmSequence:
        if isinstance(continuation_ids, torch.Tensor):
            continuation_ids = continuation_ids.tolist()
        atoms = self.tokenizer.decode_atoms(list(continuation_ids))
        rhythm = self.schema.decode(atoms)
        if len(rhythm.cycles) != expected_num_cycles:
            raise ValueError(f"Expected {expected_num_cycles} cycles, got {len(rhythm.cycles)}")
        return rhythm

    def save_derbake(self, rhythm: RhythmSequence, path: str | Path) -> None:
        write_derbake(rhythm, path)

    def save_pretrained(self, save_directory, **kwargs):
        saved = super().save_pretrained(save_directory, **kwargs)
        path = Path(save_directory) / "processor_config.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data.update({"processor_class": self.__class__.__name__,
                     "dataset_variant": self.dataset_variant,
                     "max_beats_per_cycle": self.max_beats_per_cycle,
                     "auto_map": {"AutoProcessor": "processing_ldm.LDMDerbakeProcessor"}})
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return saved

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        revision = kwargs.get("revision")
        trust_remote_code = kwargs.get("trust_remote_code", False)
        path = cached_file(pretrained_model_name_or_path, "processor_config.json",
                           revision=revision, token=kwargs.get("token"))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                                   revision=revision,
                                                   trust_remote_code=trust_remote_code,
                                                   token=kwargs.get("token"))
        return cls(tokenizer, data["dataset_variant"], data["max_beats_per_cycle"])
