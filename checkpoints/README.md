# checkpoints/

Trained LoRA adapters, produced by `train_sft.py`. The weights aren't committed to git: they don't
belong in git history. Re-running the same config gets a comparable checkpoint, not necessarily a
bit-identical one — GPU training isn't fully deterministic even with a fixed seed. This file and
each run's `run_metadata.json` are the exceptions to `.gitignore`, so a clone still shows what runs
happened even though the weights themselves don't come with it.

## Directory naming

Each run gets its own timestamped subdirectory, so repeat runs never overwrite each other:

```
checkpoints/<YYYYMMDD_HHMMSS>_<config-name>/
```

e.g. `checkpoints/20260823_151044_lora/`, produced by `configs/train/lora.yaml`
(`output_dir: checkpoints/lora`, timestamp prefixed at runtime by `timestamped_output_dir()` in
`train_sft.py`).

## Checkpoints

`train_sft_main` writes a `run_metadata.json` into the checkpoint directory right after saving the
adapter — that file is the source of truth (base model, training data, row count, epochs, LoRA
config, git commit, timestamp). This table is a human-readable copy of it; add a row here when you
add a checkpoint.

| Checkpoint | Base model | Trained on | Config | Metadata |
|---|---|---|---|---|
| `20260823_151044_lora` | Qwen2.5-3B-Instruct | `data_v1_pilot` (50 rows), 3 epochs | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20260823_151044_lora/run_metadata.json) |

**After a training run:**
1. Nothing to do for `run_metadata.json` — `train_sft.py` writes it automatically.
2. Add a row to the table above by copying values straight from that file:
   ```bash
   cat checkpoints/<new-timestamp>_lora/run_metadata.json
   ```
   Copy `base_model`, `trained_on` (+ row count/epochs), and link `train_config` and the metadata
   file.

## Re-running this recipe

Same data, config, and seed:

```bash
PYTHONPATH=src python -m grounded_refusal.train.train_sft \
  --train-config configs/train/lora.yaml
```

Then evaluate it with `--adapter checkpoints/<the-run-you-just-made>` on
`inference/run_inference.py`.
