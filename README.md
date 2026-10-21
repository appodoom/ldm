# LDM

LDM trains rhythm models from five-line `.derbake` files. It supports
`with_subd` and `no_subd` models and can package trained models for Hugging Face.

## Train in Colab

Open [the training notebook](notebooks/ldm_training.ipynb) and run its setup
cells. You need access to the private GitHub repository and the
[LDM dataset](https://huggingface.co/datasets/Appodoom/ldm-data) on Hugging Face.

1. Add your experiments to the notebook's `RUN_CONFIGS` list. Each entry sets
   its data variant, model, and training settings.
2. Run **Prepare** to build or reuse the dataset for each entry.
3. Run **Start** to train every entry in order. Progress prints in Colab.
4. To work with a saved model, set `SELECT_RUN_ID` in section 6. Then run only
   the action you need: resume, inspect metrics, generate examples, or release.

The notebook creates `MyDrive/LDM_FINAL` if needed. Runs and logs go under
`LDM_FINAL/runs`; inference outputs and release records have their own folders there.
Training does not upload a model. The release cell uploads only when you set
`PUBLISH_TO_HF = True`.

## Run locally

Requires Python 3.10 or newer. From this repository:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Files

- `ldm/`: data processing, model, training, evaluation, and Hugging Face code.
- `notebooks/ldm_training.ipynb`: Colab training and release workflow.
- `migrate_derbake.py`: converts old four-line files to five-line files.
- `tests/`: repository checks.

Each run saves `training.log`, `last.pt` for resuming, `best.pt` selected by
validation loss, and `metrics.json` with final test results. Prepared datasets
and local Hugging Face snapshots are temporary Colab files; Drive holds the
run checkpoints. Audio rendering uses the optional `audio` dependencies; install them
locally with `python -m pip install -e ".[audio]"`.

For notebook WAV previews, place samples in `MyDrive/LDM_FINAL/sounds/normal_hits/`
and `MyDrive/LDM_FINAL/sounds/decorated_hits/`. Each root contains the relevant
`doums`, `taks`, `tiks`, or `pa2s` folders. Enable `RENDER_AUDIO` in section 11.

Section 11 also has separate optional embedding reports and model/GPU checks.
Section 8 saves the test-accuracy comparison plot. Install plotting dependencies
locally with `python -m pip install -e ".[analysis]"`.

Prepared data now records checksums for every array. Rebuild older caches with
section 4. Training checks attention cache equivalence and causality before updates.
Older run checkpoints keep their original dataset fingerprint and need a verified
dataset migration before resuming under this integrity format; inference still loads them.

CUDA training tests Flash Attention forward/backward at startup, then forces it for
global attention in the verified mixed-precision dtype. The result is saved in
`attention_backend.json`. Other GPUs fall back to automatic SDPA if the test fails;
an A100 verification failure stops training. Structured attention, FP32 evaluation,
and cached decoding use automatic SDPA. CUDA profiles record this policy and probe.

`TrainConfig` also exposes `amp`, `allow_tf32`, `num_workers`, `pin_memory`,
`persistent_workers`, and `prefetch_factor`. CUDA defaults use FP16/scaling, TF32,
and two persistent batch workers with pinned memory and prefetching. CPU training
uses zero workers and disables pinning/AMP. Effective settings print at startup
and are saved in `runtime_settings.json`. Set `num_workers=0` to debug loading.
