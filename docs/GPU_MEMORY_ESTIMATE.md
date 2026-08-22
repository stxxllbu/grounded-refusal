# GPU memory estimate

Why `configs/train/sft_v1.yaml` uses `per_device_train_batch_size: 2`,
`gradient_accumulation_steps: 8`, and `gradient_checkpointing: true` on a
16GB GPU (RTX 5060 Ti). **Edit here first**, then update the config.

This is a hand estimate with the real architecture numbers plugged in, not
a measurement. See [Validating this estimate](#validating-this-estimate)
for how to check the real number on your machine.

---

## Model architecture this estimate is based on

Qwen2.5-3B-Instruct's real `config.json` (pulled from the model repo, not
assumed):

| Field | Value |
|-------|-------|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 (grouped-query attention — fewer KV heads than query heads) |
| `intermediate_size` | 11008 |
| `vocab_size` | 151936 |
| `torch_dtype` | bfloat16 |

---

## Fixed cost — independent of batch size

| Component | Estimate |
|-----------|----------|
| Weights (frozen) | 3B params x 2 bytes (bf16) = 6GB |
| LoRA weights + grad + Adam states | r=16, `target_modules=[q_proj, v_proj]`, 36 layers -> ~3.7M trainable params -> ~0.06GB |

**Fixed total: ~6GB.** LoRA's own footprint (trainable params + their
gradients + their Adam optimizer states) is under 1% of the frozen base
weights' size — this is why LoRA fine-tuning fits on hardware full
fine-tuning wouldn't.

## Full fine-tuning vs. LoRA: memory comparison

A parameter costs memory for a gradient and an Adam optimizer state only if
it is actually being trained (`requires_grad=True`). Full fine-tuning
trains all 3B parameters; LoRA trains only the small LoRA matrices and
freezes the rest. The per-trained-parameter cost is the same rule in both
cases — what differs is how many parameters that rule gets applied to.

**Cost per trained parameter** (standard mixed-precision training setup):

| Component | Bytes |
|---|---|
| Weight itself (bf16) | 2 |
| Gradient (bf16) | 2 |
| Adam momentum (fp32) | 4 |
| Adam variance (fp32) | 4 |
| fp32 master weight copy (for numerically stable optimizer updates) | 4 |
| **Total per trained parameter** | **16** |

(This is the source of the commonly cited "~16 bytes per trained
parameter" full-fine-tuning rule of thumb — sometimes simplified to ~12
without the fp32 master copy, or loosely shorthanded as "~4x" when only
referring to the weight-storage multiplier.)

**Applying it to this 3B model:**

| | Trained params | Trained-param memory | Frozen weight memory | Total |
|---|---|---|---|---|
| Full fine-tuning | 3,000,000,000 | 3B x 16 bytes ~= 48GB | 0 (nothing frozen) | **~48GB** |
| LoRA (this config) | ~3,700,000 | 3.7M x 16 bytes ~= 0.06GB | 6GB (2 bytes x 3B) | **~6.06GB** |

Full fine-tuning doesn't fit on this 16GB card, and wouldn't fit on a 24GB
card either.

---

## Activation memory — scales with batch size

Worked for `batch=4`, `seq_len=512` (the example used throughout this
doc and in `max_seq_length: 512` in the config).

**Per layer, element counts (`batch x seq_len x dim`), before converting to bytes:**

| Term | Shape | Elements |
|------|-------|----------|
| Layer input / residual stream | 4 x 512 x 2048 | 4,194,304 |
| Q projection output | 4 x 512 x 2048 (16 heads x 128 head_dim) | 4,194,304 |
| K projection output | 4 x 512 x 256 (2 KV heads x 128 head_dim — GQA) | 524,288 |
| V projection output | 4 x 512 x 256 | 524,288 |
| Attention output (post weighted-sum) | 4 x 512 x 2048 | 4,194,304 |
| MLP intermediate | 4 x 512 x 11008 | 22,544,384 |
| **Sum per layer** | | **36,175,872** |

`36,175,872 elements x 36 layers = 1,302,331,392 elements`
`x 2 bytes (bf16) = 2,604,662,784 bytes ~= 2.43GB`

(GB here is binary GiB, ÷1024^3 — matching what `nvidia-smi` and
`torch.cuda.max_memory_allocated()` report. The Fixed cost section above
used decimal GB, ÷10^9, where `3B x 2 bytes` comes out to exactly `6.0GB`.
The two conventions differ by ~7% for the same byte count; not corrected
for here, doesn't change any conclusion in this doc.)

Flash attention avoids materializing the full `seq_len x seq_len` attention
score matrix (which would otherwise be an additional, and much larger,
per-layer term) — this is why the per-layer terms above are all linear in
`seq_len`, with no `seq_len^2` term. A hand calculation that included that
matrix would substantially overestimate this section.

**LM head logits tensor** (a separate term, computed once at the end, not
per-layer): `batch x seq_len x vocab_size = 4 x 512 x 151936 = 311,164,928
elements x 2 bytes ~= 0.58GB`. Called out separately because
`vocab_size=151936` is large enough that this one tensor is not negligible
next to the per-layer sum above.

**Forward-pass subtotal: `2.43GB + 0.58GB ~= 3GB`.**

This 3GB is *only* what forward propagation needs to hold. Backward
propagation needs its own additional buffers (to compute gradients through
everything forward pass computed), plus the optimizer's own temporary
buffers, plus normal PyTorch memory-allocator fragmentation. None of these
three are modeled precisely here — the estimate instead uses a commonly
cited rule of thumb that **total activation-related memory (forward +
backward + fragmentation, together) lands at roughly 2-3x the forward-only
figure** — i.e. `3GB x 2` to `3GB x 3`, **not** `3GB + 2` or `3GB + 3`. That
gives a realistic range of **~6-9GB** for activation-related memory at
batch=4.

---

## Total memory by batch size

| Batch size | Fixed | Activations (est.) | Total | Fits in 16GB? |
|------------|-------|---------------------|-------|----------------|
| 4 | ~6GB | ~6-9GB | ~12-15GB | Close to the limit, not comfortable |
| 40 (10x) | ~6GB | ~60-90GB (scales ~linearly with batch size) | ~66-96GB | No — far past 16GB, would OOM |

---

## What the config actually implements

- **`per_device_train_batch_size: 2`** — safer than 4, given batch=4 is
  already close to the 16GB limit by this estimate.
- **`gradient_accumulation_steps: 8`** — accumulates gradients over 8
  mini-batches of 2 before each optimizer update, giving the same
  gradient-quality benefit as a true batch of 2 x 8 = 16, without ever
  holding 16 examples' activations in memory at the same time — only 2
  examples' worth of activations exist in memory at any point.
- **`gradient_checkpointing: true`** — trades compute time for activation
  memory: instead of storing all per-layer activations for the backward
  pass, it discards them after the forward pass and recomputes them on
  demand during backward. This directly reduces the ~6-9GB activation
  bucket above, which is the least certain and most GPU-limiting number in
  this estimate.

---

## Validating this estimate

This estimate uses real architecture numbers but an approximate 2-3x
backward-pass/fragmentation multiplier that isn't derived here. The
reliable way to confirm actual usage: run one training step and check
`torch.cuda.max_memory_allocated()` or `nvidia-smi`. If actual usage turns
out lower than this estimate, `per_device_train_batch_size` can be raised
back toward 4 with `gradient_accumulation_steps` reduced to match, keeping
the same effective batch size of 16.

---

## References

- Qwen2.5-3B-Instruct `config.json`: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/config.json
- Flash attention avoiding the O(seq_len^2) attention-score materialization:
  Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness" — https://arxiv.org/abs/2205.14135
