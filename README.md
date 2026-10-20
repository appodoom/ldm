# LDM rhythm model

This repository contains the five-line corpus conversion, atomic rhythm model,
training pipeline, and Hugging Face snapshot packaging. It does not train or
upload a production model automatically.

## Layout

- `ldm/`: one shared model core, rhythm codec and schemas, training, and HF classes.
- `migrate_derbake.py`: conversion of four-line source files to five-line copies.
- `tests/`: format, grammar, split, and recovery contracts.
- `notebooks/`: the full Colab training, review, upload, and remote validation notebook.

The converted corpora, prepared arrays, runs, and snapshots are ignored by Git.
The paired five-line corpus is in the private
[Appodoom/ldm-data dataset](https://huggingface.co/datasets/Appodoom/ldm-data),
under `five-line/normal` and `five-line/decorated`.

## Environment

The package requires Python 3.10 or newer and a supported PyTorch installation.
Run the notebook tests on the chosen Colab runtime before training.
The GitHub repository and HF dataset are private, so authenticate to both
services in a fresh Colab session. Clone
`https://github.com/appodoom/ldm.git` into `/content/ldm` before opening its
notebook. Install this package from the checkout:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The package pins Transformers 4.51.3 for its custom AutoClass integration.
Pure Python contracts, all converted corpus round trips, and both complete
dataset preparations passed locally. A one-epoch `with_subd` run and isolated
HF snapshot validation passed in a Python 3.13 Colab session. Production training,
`no_subd` runtime validation, generation review, and remote Hub validation remain.

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

The recommended Colab entry point is [`notebooks/ldm_training.ipynb`](notebooks/ldm_training.ipynb).
Its named `RunConfig` presets use package enums for variants, attention,
activation and position encoding. `prepare_or_load_dataset` hashes the full
recipe and source bytes; `start_run` creates a unique Drive run ID; inspect and
resume are separate actions. New run folders contain `experiment.json`, live
`training.log`, `last.pt`, `best.pt`, `evaluations.jsonl`, `metrics.json` and
optional reports. The older fixed-path CLI remains for existing runs.

Experiment controls include validation cadence, evaluation batch size, AdamW
weight decay, CUDA AMP, gradient clipping, early stopping and explicit step,
time, loss and divergence stops. The selected checkpoint is chosen only by
validation loss; final test metrics use that checkpoint. Reports use unique
scored targets and include ordinary/decorated, HIT, structure, decoration,
skeleton and (for `with_subd`) hit-count measures. Optional package functions
generate review files, WAV previews from supplied samples, embedding neighbors,
contextual PCA tables, plots and comparison CSVs. The WAV renderer needs four
mono 16-bit WAV sample folders at its selected sample rate.

This workflow has pure Python contract coverage locally. Its Torch training,
resume and HF snapshot paths must be exercised in Colab on the pinned dataset
before a production release. Existing Drive runs are not migrated or overwritten.

The older direct CLI command is still available:

```bash
python -m ldm.cli train --dataset datasets/with_subd --run runs/with_subd \
  --epochs 10 --batch-size 4 --learning-rate 0.0003
```

`last.pt` contains optimizer, scaler, RNG, and epoch/batch position for resume;
`best.pt` contains the selected weights and model configuration. Continue with
`--resume`. The run also stores its config and dataset reference for promotion.
Training prints live loss every 50 optimizer steps, checkpoint saves, and
validation loss at each epoch end. Change the interval with
`--log-every-steps N`; use `python -u -m ldm.cli train ...` in Colab to stream it.
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
