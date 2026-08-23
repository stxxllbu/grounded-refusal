# checkpoints/

Trained LoRA adapters, produced by `train_sft.py`. Not committed to git --
binary model weights don't belong in git history, and are cheap to
reproduce from the training script + config + data.

This file is the one exception (see `.gitignore`): it exists so the
directory itself survives a clone, and so it's clear the directory being
empty is intentional, not a missing upload.

## Layout

Each run gets its own timestamped subdirectory, so repeat runs never
overwrite each other:

```
checkpoints/<YYYYMMDD_HHMMSS>_<config-name>/
```

e.g. `checkpoints/20260823_151044_lora/`, produced by
`configs/train/lora.yaml` (`output_dir: checkpoints/lora`, timestamp
prefixed at runtime by `timestamped_output_dir()` in `train_sft.py`).

## Reproducing a checkpoint

```bash
PYTHONPATH=src python -m grounded_refusal.train.train_sft \
  --train-config configs/train/lora.yaml
```

Then evaluate it with `--adapter checkpoints/<the-run-you-just-made>` on
`inference/run_inference.py`.
