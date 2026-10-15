# LDM rhythm model

This repository contains the five-line corpus conversion, atomic rhythm model,
training pipeline, and Hugging Face snapshot packaging. It does not train or
upload a production model automatically.

## Layout

- `ldm/`: one shared model core, rhythm codec and schemas, training, and HF classes.
- `migrate_derbake.py`: conversion of four-line source files to five-line copies.
- `tests/`: format, grammar, split, and recovery contracts.
- `notebooks/`: a Colab orchestration notebook that imports this package.

The converted corpora, prepared arrays, runs, and snapshots are ignored by Git.
The paired five-line corpus is in the private
[Appodoom/ldm-data dataset](https://huggingface.co/datasets/Appodoom/ldm-data),
under `five-line/normal` and `five-line/decorated`.

## Environment

Use Python 3.10–3.12 on a machine with a supported PyTorch installation.
The GitHub repository and HF dataset are private, so authenticate to both
services in a fresh Colab session. Clone this repository before opening its
notebook. Install this package from the checkout:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The package pins Transformers 4.51.3 for its custom AutoClass integration.
In the development workspace, pure Python contracts, all converted corpus round
trips, and both complete dataset preparations passed. Model training and HF
snapshot commands still need validation on a machine with PyTorch and
Transformers before any release is published.

## Prepare a dataset

Prepare either variant from the ordinary corpus, optionally including its
decorated partner. Every file is split by complete cycles into approximately
70% training, 20% validation, and 10% testing. The ordinary/decorated versions
share the same boundaries. Chunks never cross those boundaries.

Authenticate to Hugging Face to access the private dataset. Download the exact
five-line corpus revision used here:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Appodoom/ldm-data",
    repo_type="dataset",
    revision="34fb2449a25472e856a4db0aa85e280e9a82a8b2",
    allow_patterns="five-line/**",
    local_dir=".derbake",
)
```

```bash
python -m ldm.cli prepare --ordinary .derbake/five-line/normal --decorated .derbake/five-line/decorated \
  --variant with_subd --context-size 1024 --max-beats-per-cycle 16 \
  --output datasets/with_subd
python -m ldm.cli prepare --ordinary .derbake/five-line/normal --decorated .derbake/five-line/decorated \
  --variant no_subd --context-size 1024 --max-beats-per-cycle 16 \
  --output datasets/no_subd
```

Omit `--decorated` to use only the ordinary corpus. `dataset_manifest.json`
records the per-file cycle ranges and source fingerprint. It is invalidated by
changed source files, variant, chunking, or split rules.
The default stride is half the context size, so complete subdivision events
remain in an overlapping chunk. `--decorated-weight` changes the decorated
examples' training sampling weight; validation and test remain unweighted.

## Train later

```bash
python -m ldm.cli train --dataset datasets/with_subd --run runs/with_subd \
  --epochs 10 --batch-size 4 --learning-rate 0.0003
```

`last.pt` contains optimizer, scaler, RNG, and epoch/batch position for resume;
`best.pt` contains the selected weights and model configuration. Continue with
`--resume`. The run also stores its config and dataset reference for promotion.
New atomic models start from fresh weights; historic notebook
checkpoints are not silently reindexed.

## Build and validate a local HF snapshot later

```bash
python -m ldm.cli snapshot --run runs/with_subd --output snapshots/with_subd
python -m ldm.cli validate --snapshot snapshots/with_subd
```

This writes model weights, custom code, tokenizer, processor, manifest, and model
card to one folder. It does not upload. `validate` starts an isolated Python process
and checks `AutoModel`, `AutoTokenizer`, `AutoProcessor`, a forward pass, and a
`.derbake` round trip. Only a validated snapshot should later be uploaded and
tested again from a pinned HF commit. The reserved private model repositories
are [ldm-with-subd](https://huggingface.co/Appodoom/ldm-with-subd) and
[ldm-no-subd](https://huggingface.co/Appodoom/ldm-no-subd); they contain no
trained weights yet.

## Public inference contract

`AutoProcessor` parses a strict five-line `.derbake` prompt. `AutoModel.generate`
samples normally until it accepts `num_cycles` valid new cycles, discarding and
reporting malformed sampled cycles. It returns only those accepted continuation
IDs. A protective `max_new_tokens` cap causes an explicit incomplete error if
too few valid cycles were accepted. The processor writes only generated cycles
with 120 BPM, amplitude 1.5, and a blank skeleton line.
