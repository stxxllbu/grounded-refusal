# Preference generation protocol

How we build preference JSONL from approved QA rows: `data/preference_v1_pilot.jsonl`, and later `data/preference_v1.jsonl`.

**QA construction (must be done first):** [`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md).  
**QA labels:** [`DATA_LABELS.md`](DATA_LABELS.md).  
**Schema:** [`src/data/schema_pref.py`](../src/data/schema_pref.py) (`PreferencePair`, `NegativeType`).

This does **not** rebuild QA. Each preference row is **derived** from one QA row via `base_example_id`.

---

## Principle

1. Start from an **approved** QA row (pilot: `data_v1_pilot.jsonl`).
2. Choose `negative_type` from that row’s `answerability` × `evidence_challenge` (fixed map; not LLM-chosen).
3. Assemble `prompt`; set `chosen` = QA `reference_answer`.
4. Write one deliberate bad answer as `rejected` (API or human), matching that `negative_type`.
5. Pilot default: **1 QA → 1 preference pair**. More pairs per QA are optional later.

Do not invent a new `answerability`. Slice with existing QA labels:

- `answerability` = correct behavior  
- `evidence_challenge` = why the item is hard  
- `negative_type` = how `rejected` fails  

---

## Fields

One row = one DPO training pair.

| Field | Required | Purpose |
|-------|----------|---------|
| `id` | yes | Preference id, e.g. `pref_0021` (optional letter suffix for multiple pairs: `pref_0021a`). |
| `base_example_id` | yes | Links to QA `id`, e.g. `ex_0021`. |
| `prompt` | yes | Full model input assembled from the source QA row and prompt config. |
| `chosen` | yes | Preferred answer (= QA `reference_answer`). |
| `rejected` | yes | One specific bad answer. |
| `negative_type` | yes | Failure mode of `rejected` (see map below). |
| `dataset_version` | yes | Same family as QA, e.g. `v1`. |
| `metadata` | no | e.g. `creation_process`, `notes`. |

### How `prompt` is constructed

The builder deterministically assembles `prompt` from:

- `evidence` and `question` in the source QA row;
- `instruction`, `evidence_label`, and `question_label` in
  [`configs/prompts/default.yaml`](../configs/prompts/default.yaml).

The resulting format is:

```text
{evidence_label}:
{QA evidence}

{question_label}:
{QA question}

Instruction:
{instruction}
```

No LLM generates or rewrites `prompt`. The LLM is used only to generate the
`rejected` response.

`NegativeType` enum values in schema (pilot uses all five):  
`hallucination` | `over_refusal` | `over_complete` | `distractor_confusion` | `memory_override`.

---

## Pipeline

```text
approved QA JSONL
        ↓
choose negative_type  (Python map)
        ↓
assemble prompt + chosen
        ↓
generate rejected     (API writes text only)
        ↓
PreferencePair validate
        ↓
preference_*.jsonl
```

| Step | What it does | Where |
|------|----------------|-------|
| Input QA | Approved Layer 2 (or full) QA rows | e.g. `data/data_v1_pilot.jsonl` |
| Map type | `answerability` × `evidence_challenge` → `negative_type` | [`build_preference.py`](../src/data/build_preference.py) `choose_negative_type` |
| Prompt / chosen | Format prompt; copy `reference_answer` | same script + `configs/prompts/default.yaml` |
| Rejected | API generates bad answer for fixed type | `generate_rejected` in `build_preference.py` |
| Validate | Field shape + enum | `PreferencePair` in [`schema_pref.py`](../src/data/schema_pref.py) |
| IO | Read/write JSONL | [`util/io.py`](../src/util/io.py) |
| Output | Preference dataset | e.g. `data/preference_v1_pilot.jsonl` |

Python decides `negative_type` and system instructions; the API returns **only** `rejected` text. Details: **FAQ**.

---

## Construction rules

Fields lists every column. Fill order:

1. **Pick QA row** — must already satisfy [`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md).
2. **Choose `negative_type`** — pilot map below (required).
3. **Assemble locals** — `id`, `base_example_id`, `prompt`, `chosen`, `dataset_version`.
4. **Write `rejected`** — wrong for that `negative_type`, similar fluency to `chosen`.
5. **Optional metadata** — e.g. `creation_process: llm_generated`.

### Pilot negative map (required)

`answerability` × `evidence_challenge` → one `negative_type`.

| # | `answerability` | `evidence_challenge` | Correct (`chosen`) | Bad (`rejected`) | `negative_type` |
|---|-----------------|----------------------|--------------------|------------------|-----------------|
| 1 | `answerable` | `[]` | Answer from evidence | Refuse even though evidence answers | `over_refusal` |
| 2 | `answerable` | `["known_world_conflict"]` | Answer from evidence | Prefer world knowledge over evidence | `memory_override` |
| 3 | `unanswerable` | `[]` | Refuse | Fabricate a plausible answer | `hallucination` |
| 4 | `unanswerable` | `["distractor_entity"]` | Refuse | Answer using distractor / wrong entity | `distractor_confusion` |
| 5 | `partial` | `["partial_evidence"]` | Answer supported; refuse unsupported | Also fill unsupported part | `over_complete` |

Decision order (same as code):

1. `answerable` + `known_world_conflict` → `memory_override`
2. Else `answerable` → `over_refusal`
3. Else `unanswerable` + `distractor_entity` → `distractor_confusion`
4. Else `unanswerable` → `hallucination`
5. Else `partial` → `over_complete`

Notes:

- `known_world_conflict` is **answerable**, not unanswerable.
- Do not use one generic hallucination for all unanswerable rows: `[]` vs `distractor_entity` need different `rejected` content.
- Pilot `partial`: **`over_complete` only**.

### Bookkeeping

| Field | How to set |
|-------|------------|
| `id` | `pref_` + digits from QA id (`ex_0021` → `pref_0021`) |
| `base_example_id` | QA `id` |
| `dataset_version` | Match QA (e.g. `v1`) |
| `metadata.creation_process` | typically `llm_generated` when API writes `rejected` |

### Optional later negatives (not required for pilot)

| QA situation | Extra `rejected` idea | `negative_type` |
|--------------|----------------------|-----------------|
| `partial` | Refuse the whole question | `over_refusal` |
| `answerable` + `known_world_conflict` | Second pair with the other of `{memory_override, over_refusal}` | as labeled |
| any | Style-only bad answers / typos | avoid — not a grounding failure |

---

## Scale and pilot

| Split | Preference rows |
|-------|-----------------|
| Pilot `preference_v1_pilot.jsonl` | **50** (1:1 with `data_v1_pilot.jsonl`) |
| Full `preference_v1.jsonl` | ~**500** (1:1 with `data_v1.jsonl`), after full QA exists |

Distributions: [`reports/week2.md`](reports/week2.md).

**Workflow**

1. Freeze / approve the QA file used as base.
2. Run [`build_preference.py`](../src/data/build_preference.py) (map type → API `rejected` → validate → write).
3. Human-review a sample across all five slices (especially `memory_override`).
4. Scale with the same map when full QA exists.

---

## FAQ

### Does the API choose `negative_type`?

No. Python chooses it from the pilot map, then picks the matching system instruction. The API only returns `rejected` text.

### Why not store `evidence` / `answerability` again on the preference row?

They live on the QA row. Preference links with `base_example_id` and puts the training input in `prompt`. Avoids duplicated drifting copies.

### Style constraints for `chosen` / `rejected`?

- Similar fluency (avoid template `chosen` vs polished `rejected`, or DPO may reward style).
- Prefer QA that already finished Layer 2 paraphrase.
- `rejected` must be wrong **for the labeled `negative_type`**, not just a paraphrase of `chosen`.

### What does schema validation cover?

`PreferencePair(...)` checks field shape and that `negative_type` is one of the five enums. It does **not** judge whether the `rejected` text truly matches that failure mode—that needs human review (or a later judge).
