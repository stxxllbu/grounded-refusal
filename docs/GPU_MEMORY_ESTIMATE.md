# GPU memory estimate

Why `configs/train/sft_v1.yaml` uses `per_device_train_batch_size: 2`,
`gradient_accumulation_steps: 8`, and `gradient_checkpointing: true` on a
16GB GPU (RTX 5060 Ti). **Edit here first**, then update the config.

This is a hand estimate, not a measurement — see [Validating this
estimate](#validating-this-estimate) at the end for how to check the real
number.

---

## Model architecture this estimate is based on

Qwen2.5-3B-Instruct's real `config.json` (not assumed — pulled from the
model repo):

| Field | Value |
|-------|-------|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 (grouped-query attention) |
| `intermediate_size` | 11008 |
| `vocab_size` | 151936 |
| `torch_dtype` | bfloat16 |

---

## Fixed cost — independent of batch size

| Component | Estimate |
|-----------|----------|
| Weights (frozen) | 3B params x 2 bytes (bf16) ~= 6GB |
| LoRA weights + grad + Adam states | r=16, `target_modules=[q_proj, v_proj]`, 36 layers -> ~3.7M trainable params -> ~0.05GB |

**Fixed total: ~6GB.** LoRA's own footprint is negligible relative to the
frozen base weights — this is the whole point of using LoRA instead of full
fine-tuning (see [`PREFERENCE_GENERATION_PROTOCOL.md`](PREFERENCE_GENERATION_PROTOCOL.md)-adjacent
discussion of parameter-efficient fine-tuning if unfamiliar).

---

## Activation memory — scales with batch size, batch=4 / seq_len=512 example

Per transformer layer, summed across the Q/K/V projections, attention
output, and MLP intermediate activations (`batch x seq_len x dim` for each
term), then multiplied across 36 layers: **~2.4GB**.

Flash attention avoids materializing the full `seq_len^2` attention score
matrix, so this stays close to linear in `batch x seq_len` rather than
quadratic — a naive hand calculation that includes the full attention score
matrix would overestimate this term substantially.

Plus the LM head logits tensor (`batch x seq_len x vocab_size` — and
`vocab_size=151936` is large enough that this is not negligible): **~0.6GB**.

Forward-pass-only subtotal: **~3GB**. In practice, backward-pass buffers,
optimizer temporary buffers, and allocator fragmentation add roughly
**2-3x** on top of this (not modeled precisely here — see [Validating this
estimate](#validating-this-estimate)), giving a realistic range of
**~6-9GB** of activation memory at batch=4.

---

## Total, and why batch=40 was rejected

| Batch size | Fixed | Activations (est.) | Total | Fits in 16GB? |
|------------|-------|---------------------|-------|----------------|
| 4 | ~6GB | ~6-9GB | ~12-15GB | Close to the limit, not comfortable |
| 40 (10x) | ~6GB | ~60-90GB (activations scale ~linearly with batch) | ~66-96GB | No — far past 16GB, would OOM |

---

## What the config actually implements

- **`per_device_train_batch_size: 2`** — safer than 4, given batch=4 is
  already close to the 16GB limit by this estimate.
- **`gradient_accumulation_steps: 8`** — recovers the gradient-quality
  benefit of a larger effective batch (2 x 8 = 16) without ever holding 16
  examples' activations in memory at once; see
  [`EVAL_METRICS.md`](EVAL_METRICS.md) for the same "keep the expensive part
  small, do more of the cheap part" pattern applied elsewhere in this repo.
- **`gradient_checkpointing: true`** — trades compute time for activation
  memory by recomputing activations during the backward pass instead of
  storing all of them; this directly reduces the ~6-9GB activation bucket,
  which is the least certain and most GPU-limiting number in this estimate.

---

## Validating this estimate

This is a hand estimate with real architecture numbers but an approximate
2-3x backward-pass/fragmentation multiplier. The reliable way to confirm
actual usage: run one training step and check
`torch.cuda.max_memory_allocated()` or `nvidia-smi`. If it turns out there's
more headroom than estimated, `per_device_train_batch_size` can be raised
back toward 4 with `gradient_accumulation_steps` reduced to match, keeping
the same effective batch size of 16.

---

## References

- Qwen2.5-3B-Instruct `config.json`: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/config.json
- Flash attention avoiding the O(seq_len^2) attention-score materialization:
  Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness" — https://arxiv.org/abs/2205.14135
