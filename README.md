# LDM

Train and use rhythm models with five-line `.derbake` files.
Supports `with_subd` and `no_subd`.

## Use in Colab

Open [the notebook](notebooks/ldm_training.ipynb).

1. Add `GITHUB_TOKEN` to Colab Secrets and allow notebook access.
2. Run setup, sign in to Hugging Face, select data, and mount Drive.
3. Define experiments in `RUN_CONFIGS`. Each chooses a subdivision variant and an independent normal/decorated source.
4. Run dataset preparation, then start training.
5. Set `SELECT_RUN_ID` to resume a run, view results, or generate rhythms.
6. To publish a model, use the release section and set `PUBLISH_TO_HF = True`.

Runs are saved in `MyDrive/LDM_FINAL/runs`. Each run contains:

- `last.pt`: resume checkpoint.
- `best.pt`: best model by validation loss.
- `metrics.json`: evaluation results.
- `training.log`: progress logs.

## Local setup

Requires Python 3.10 or newer.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The implementation is in `ldm/`; the checks are in `tests/`.
