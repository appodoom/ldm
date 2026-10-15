"""Build a coherent local HF snapshot from a new atomic checkpoint."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from .configuration_ldm import LDMConfig
from .modeling_ldm import LDMModel
from .processing_ldm import LDMDerbakeProcessor
from .tokenization_ldm import LDMTokenizer
from .vocabulary import atomic_vocabulary, vocabulary_hash


def build_local_snapshot(run_dir: str | Path, output_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    config = LDMConfig(**checkpoint["config"])
    vocab = atomic_vocabulary(config.dataset_variant)
    if config.vocab_size != len(vocab) or config.vocab_hash != vocabulary_hash(vocab):
        raise ValueError("Checkpoint vocabulary does not match the atomic schema")
    model = LDMModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    tokenizer = LDMTokenizer(tokens=list(vocab))
    processor = LDMDerbakeProcessor(tokenizer, config.dataset_variant,
                                    config.max_beats_per_cycle)
    LDMConfig.register_for_auto_class("AutoConfig")
    LDMModel.register_for_auto_class("AutoModel")
    LDMTokenizer.register_for_auto_class("AutoTokenizer")
    LDMDerbakeProcessor.register_for_auto_class("AutoProcessor")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    reloaded = LDMModel.from_pretrained(output_dir)
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), reloaded.state_dict()[name].cpu()):
            raise RuntimeError(f"Snapshot changed checkpoint tensor {name}")
    payload = {"source_checkpoint_sha256": hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest(),
               "dataset_fingerprint": checkpoint["dataset_fingerprint"],
               "dataset_variant": config.dataset_variant, "vocab_hash": config.vocab_hash,
               "derbake_format_version": config.derbake_format_version,
               "token_schema_version": config.token_schema_version,
               "source_epoch": checkpoint["epoch"], "source_val_loss": checkpoint["val_loss"]}
    metadata = output_dir / "metadata"
    metadata.mkdir()
    for name in ("run_config.json", "dataset_ref.json"):
        shutil.copyfile(run_dir / name, metadata / name)
    (metadata / "source_run.json").write_text(json.dumps({
        "epoch": checkpoint["epoch"], "validation_loss": checkpoint["val_loss"],
        "checkpoint_sha256": payload["source_checkpoint_sha256"]}, indent=2) + "\n",
        encoding="utf-8")
    (metadata / "release_manifest.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                    encoding="utf-8")
    repo_id = {"with_subd": "Appodoom/ldm-with-subd",
               "no_subd": "Appodoom/ldm-no-subd"}[config.dataset_variant]
    (output_dir / "README.md").write_text(
        f"""---
library_name: transformers
tags:
- rhythm-generation
- derbake
---

# LDM {config.dataset_variant}

This snapshot uses atomic rhythm tokens and five-line `.derbake` files.
It was built from a selected training checkpoint. `trust_remote_code=True`
executes the model/processor code from the selected repository revision.

```python
from transformers import AutoModel, AutoProcessor

repo_id = "{repo_id}"
revision = "main"  # Replace with the verified release commit SHA for reproducibility.
processor = AutoProcessor.from_pretrained(repo_id, revision=revision, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, revision=revision, trust_remote_code=True).eval()
inputs = processor(derbake_path="prompt.derbake", return_tensors="pt")
ids = model.generate(input_ids=inputs["input_ids"].to(next(model.parameters()).device),
                     num_cycles=10, max_new_tokens=4096)
rhythm = processor.decode_generation(ids[0], expected_num_cycles=10)
processor.save_derbake(rhythm, "generated.derbake")
```

The call returns only newly accepted cycles. Invalid sampled cycles are skipped
with a Python warning; use `return_generation_report=True` for details. A token
cap reached before ten valid cycles raises an error and produces no output file.
The writer uses 120 BPM, amplitude 1.5, and a blank skeleton line. The model
supports batch size one and at most {config.max_beats_per_cycle} beats per cycle.
""", encoding="utf-8")
    return payload
