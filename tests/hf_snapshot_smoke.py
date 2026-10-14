"""Run with `python -I` to prove a snapshot works without the source package."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer


def main(snapshot: Path) -> None:
    model = AutoModel.from_pretrained(snapshot, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    if model.config.vocab_hash != tokenizer.atomic.hash or tokenizer.atomic.hash != processor.tokenizer.atomic.hash:
        raise AssertionError("Model, tokenizer, and processor vocabularies differ")

    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "prompt.derbake"
        source.write_text("120\n120 120 120 120\n4\n\n"
                          "SUBD_1 HIT_D AMP_1 SUBD_1 HIT_S AMP_1 "
                          "SUBD_1 HIT_OTA AMP_1 SUBD_1 HIT_S AMP_1\n", encoding="utf-8")
        original = processor.schema.decode(processor.tokenizer.decode_atoms(
            processor(derbake_path=source)["input_ids"][0]))
        encoded = processor(derbake_path=source, return_tensors="pt")
        target = Path(folder) / "roundtrip.derbake"
        processor.save_derbake(original, target)
        if processor(derbake_path=target)["input_ids"] != processor(derbake_path=source)["input_ids"]:
            raise AssertionError("Processor file round trip changed model atoms")

        routing = None
        if model.core.has_structured_layers:
            package = model.__class__.__module__.rsplit(".", 1)[0]
            rhythm = importlib.import_module(f"{package}.rhythm_ldm")
            ids = encoded["input_ids"][0].tolist()
            lookup = rhythm.build_rhythm_lookup_arrays(list(model.vocab),
                                                        max_beat_positions=model.core.max_beat_positions)
            hierarchy = rhythm.compute_hierarchy_ids_for_sequence(ids, lookup)
            arrays, _ = rhythm.build_structured_routing_for_window(hierarchy, 0, len(ids), len(ids))
            routing = {name: torch.from_numpy(value).unsqueeze(0) for name, value in arrays.items()}
        with torch.no_grad():
            logits = model(input_ids=encoded["input_ids"], structured_routing=routing).logits
        if logits.shape != (1, encoded["input_ids"].shape[1], model.config.vocab_size):
            raise AssertionError("Unexpected model output shape")
    print("Isolated HF AutoClass load, forward, and file round trip passed")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
