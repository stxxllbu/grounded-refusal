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

Runs made before 2026-10-09 do not record a seed in `run_metadata.json`; they used 42, which
was fixed in the training configs at the time. From 2026-10-09 the seed is set with `--seed`
and recorded.

| Checkpoint | Base model | Trained on | Seed | Config | Metadata |
|---|---|---|---|---|---|
| `20260823_151044_lora` | Qwen2.5-3B-Instruct | `data_v1_pilot` (50 rows), 3 epochs | 42 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20260823_151044_lora/run_metadata.json) |
| `20260913_193703_lora` | Qwen2.5-3B-Instruct | `data_v2_train` (480 rows), 3 epochs | 42 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20260913_193703_lora/run_metadata.json) |
| `20260927_150840_lora` | Qwen2.5-3B-Instruct | `data_v3a_train` (672 rows), 3 epochs | 42 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20260927_150840_lora/run_metadata.json), [`trainer_state.json`](20260927_150840_lora/trainer_state.json) |
| `20260930_145645_lora` | Qwen2.5-3B-Instruct | `data_v3b_train` (672 rows), 3 epochs | 42 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20260930_145645_lora/run_metadata.json), [`trainer_state.json`](20260930_145645_lora/trainer_state.json) |
| `20260930_153531_lora` | Qwen2.5-3B-Instruct | `data_v3b_train` (672 rows), 3 epochs | 42 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20260930_153531_lora/run_metadata.json), [`trainer_state.json`](20260930_153531_lora/trainer_state.json) |
| `20261008_223428_lora` | Qwen2.5-3B-Instruct | `data_v3c_train` (672 rows), 3 epochs | 42 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20261008_223428_lora/run_metadata.json), [`trainer_state.json`](20261008_223428_lora/trainer_state.json) |
| `20261009_164200_lora` | Qwen2.5-3B-Instruct | `data_v3a_train` (672 rows), 3 epochs | 44 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20261009_164200_lora/run_metadata.json), [`trainer_state.json`](20261009_164200_lora/trainer_state.json) |
| `20261009_171252_lora` | Qwen2.5-3B-Instruct | `data_v3c_train` (672 rows), 3 epochs | 44 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20261009_171252_lora/run_metadata.json), [`trainer_state.json`](20261009_171252_lora/trainer_state.json) |
| `20261009_174213_lora` | Qwen2.5-3B-Instruct | `data_v3a_train` (672 rows), 3 epochs | 45 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20261009_174213_lora/run_metadata.json), [`trainer_state.json`](20261009_174213_lora/trainer_state.json) |
| `20261009_181152_lora` | Qwen2.5-3B-Instruct | `data_v3c_train` (672 rows), 3 epochs | 45 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20261009_181152_lora/run_metadata.json), [`trainer_state.json`](20261009_181152_lora/trainer_state.json) |
| `20261009_184020_lora` | Qwen2.5-3B-Instruct | `data_v3a_train` (672 rows), 3 epochs | 46 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20261009_184020_lora/run_metadata.json), [`trainer_state.json`](20261009_184020_lora/trainer_state.json) |
| `20261009_190915_lora` | Qwen2.5-3B-Instruct | `data_v3c_train` (672 rows), 3 epochs | 46 | [`../configs/train/lora.yaml`](../configs/train/lora.yaml) | [`run_metadata.json`](20261009_190915_lora/run_metadata.json), [`trainer_state.json`](20261009_190915_lora/trainer_state.json) |

### DPO checkpoints

A DPO adapter is trained on top of the base model with its SFT adapter already merged in, so it only
means something together with that SFT checkpoint. `run_metadata.json` records it as `sft_adapter`,
and `run_inference.py --adapter` reads it to load the two in the right order.

| Checkpoint | Base model | Starts from (SFT adapter) | Preference data | Seed | β | Config | Metadata |
|---|---|---|---|---|---|---|---|
| `20260926_143154_dpo` | Qwen2.5-3B-Instruct | `20260913_193703_lora` | `preference_v2_train` (480 pairs), 3 epochs | 42 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20260926_143154_dpo/run_metadata.json), [`trainer_state.json`](20260926_143154_dpo/trainer_state.json) |
| `20260927_152858_dpo` | Qwen2.5-3B-Instruct | `20260927_150840_lora` | `preference_v3a_train` (672 pairs), 3 epochs | 42 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20260927_152858_dpo/run_metadata.json), [`trainer_state.json`](20260927_152858_dpo/trainer_state.json) |
| `20260930_150138_dpo` | Qwen2.5-3B-Instruct | `20260930_145645_lora` | `preference_v3b_train` (672 pairs), 3 epochs | 42 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20260930_150138_dpo/run_metadata.json), [`trainer_state.json`](20260930_150138_dpo/trainer_state.json) |
| `20260930_154022_dpo` | Qwen2.5-3B-Instruct | `20260930_153531_lora` | `preference_v3b_train` (672 pairs), 3 epochs | 42 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20260930_154022_dpo/run_metadata.json), [`trainer_state.json`](20260930_154022_dpo/trainer_state.json) |
| `20261008_223937_dpo` | Qwen2.5-3B-Instruct | `20261008_223428_lora` | `preference_v3c_train` (672 pairs), 3 epochs | 42 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20261008_223937_dpo/run_metadata.json), [`trainer_state.json`](20261008_223937_dpo/trainer_state.json) |
| `20261009_164652_dpo` | Qwen2.5-3B-Instruct | `20261009_164200_lora` | `preference_v3a_train` (672 pairs), 3 epochs | 44 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20261009_164652_dpo/run_metadata.json), [`trainer_state.json`](20261009_164652_dpo/trainer_state.json) |
| `20261009_171742_dpo` | Qwen2.5-3B-Instruct | `20261009_171252_lora` | `preference_v3c_train` (672 pairs), 3 epochs | 44 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20261009_171742_dpo/run_metadata.json), [`trainer_state.json`](20261009_171742_dpo/trainer_state.json) |
| `20261009_174703_dpo` | Qwen2.5-3B-Instruct | `20261009_174213_lora` | `preference_v3a_train` (672 pairs), 3 epochs | 45 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20261009_174703_dpo/run_metadata.json), [`trainer_state.json`](20261009_174703_dpo/trainer_state.json) |
| `20261009_181642_dpo` | Qwen2.5-3B-Instruct | `20261009_181152_lora` | `preference_v3c_train` (672 pairs), 3 epochs | 45 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20261009_181642_dpo/run_metadata.json), [`trainer_state.json`](20261009_181642_dpo/trainer_state.json) |
| `20261009_184511_dpo` | Qwen2.5-3B-Instruct | `20261009_184020_lora` | `preference_v3a_train` (672 pairs), 3 epochs | 46 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20261009_184511_dpo/run_metadata.json), [`trainer_state.json`](20261009_184511_dpo/trainer_state.json) |
| `20261009_191404_dpo` | Qwen2.5-3B-Instruct | `20261009_190915_lora` | `preference_v3c_train` (672 pairs), 3 epochs | 46 | 0.1 | [`../configs/train/dpo.yaml`](../configs/train/dpo.yaml) | [`run_metadata.json`](20261009_191404_dpo/run_metadata.json), [`trainer_state.json`](20261009_191404_dpo/trainer_state.json) |

**After a training run:**
1. Nothing to do for `run_metadata.json` and `trainer_state.json` — the training script writes them
   automatically.
2. Add a row to the table for its kind by copying values straight from `run_metadata.json`:
   ```bash
   cat checkpoints/<new-timestamp>_<lora|dpo>/run_metadata.json
   ```
   SFT: copy `base_model`, `trained_on` (+ row count/epochs), `seed`, and link `train_config` and
   the metadata files. DPO: also copy `sft_adapter` and `beta`.

## Re-running this recipe

Same data, config, and seed:

```bash
PYTHONPATH=src python -m grounded_refusal.train.train_sft \
  --data data/data_v2_train.jsonl \
  --train-config configs/train/lora.yaml
```

Then evaluate it with `--adapter checkpoints/<the-run-you-just-made>` on
`inference/run_inference.py`.
