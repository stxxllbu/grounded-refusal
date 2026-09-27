# checkpoints/

Trained LoRA adapters, produced by `train_sft.py` (SFT) and `train_dpo.py` (DPO). The weights aren't
committed to git: they don't belong in git history. Re-running the same config gets a comparable
checkpoint, not necessarily a bit-identical one — GPU training isn't fully deterministic even with a
fixed seed. This file and each run's `run_metadata.json` and `trainer_state.json` are the exceptions
to `.gitignore`, so a clone still shows what runs happened even though the weights themselves don't
come with it.

## Directory naming

Each run gets its own timestamped subdirectory, so repeat runs never overwrite each other:

```
checkpoints/<YYYYMMDD_HHMMSS>_<config-name>/
```

e.g. `checkpoints/20260823_151044_lora/`, produced by `configs/train/lora.yaml`
(`output_dir: checkpoints/lora`, timestamp prefixed at runtime by `timestamped_output_dir()` in
`train_sft.py`).

## Checkpoints

Both training scripts write a `run_metadata.json` into the checkpoint directory right after saving
the adapter — that file is the source of truth (base model, training data, row count, epochs, LoRA
config, git commit, timestamp). The tables below are human-readable copies of it; add a row to the
table for the kind of run you made.

### SFT checkpoints

| Checkpoint | Base model | Trained on | Config | Metadata |
|---|---|---|---|---|
| `20260823_151044_lora` | Qwen2.5-3B-Instruct | `data_v1_pilot` (50 rows), 3 epochs | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20260823_151044_lora/run_metadata.json) |
| `20260913_193703_lora` | Qwen2.5-3B-Instruct | `data_v2_train` (480 rows), 3 epochs | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20260913_193703_lora/run_metadata.json) |

### DPO checkpoints

A DPO adapter is trained on top of the base model with its SFT adapter already merged in, so it only
means something together with that SFT checkpoint. `run_metadata.json` records it as `sft_adapter`,
and `run_inference.py --adapter` reads it to load the two in the right order.

| Checkpoint | Base model | Starts from (SFT adapter) | Preference data | β | Config | Metadata |
|---|---|---|---|---|---|---|
| `20260926_143154_dpo` | Qwen2.5-3B-Instruct | `20260913_193703_lora` | `preference_v2_train` (480 pairs), 3 epochs | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20260926_143154_dpo/run_metadata.json), [`trainer_state.json`](20260926_143154_dpo/trainer_state.json) |

**After a training run:**
1. Nothing to do for `run_metadata.json` and `trainer_state.json` — the training script writes them
   automatically.
2. Add a row to the table for its kind by copying values straight from `run_metadata.json`:
   ```bash
   cat checkpoints/<new-timestamp>_<lora|dpo>/run_metadata.json
   ```
   SFT: copy `base_model`, `trained_on` (+ row count/epochs), and link `train_config` and the
   metadata file. DPO: also copy `sft_adapter` and `beta`.

## Re-running this recipe

Same data, config, and seed:

```bash
PYTHONPATH=src python -m grounded_refusal.train.train_sft \
  --data data/data_v2_train.jsonl \
  --train-config configs/train/lora.yaml
```

Then evaluate it with `--adapter checkpoints/<the-run-you-just-made>` on
`inference/run_inference.py`.
