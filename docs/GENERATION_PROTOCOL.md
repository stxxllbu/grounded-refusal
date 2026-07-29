# GENERATION_PROTOCOL (deprecated)

**This file is deprecated as a single combined protocol.**

| Topic | Use instead |
|-------|-------------|
| **QA** generation (fields + Layer 1/2 + construction rules) | [`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md) |
| **Preference** generation | Section below for now → later `PREFERENCE_GENERATION_PROTOCOL.md` |

Label definitions: [`DATA_LABELS.md`](DATA_LABELS.md).

Do not add new QA rules here. Edit [`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md).

---

# Preference data generation protocol

How we build `data/preference_v1_pilot.jsonl` (and later `preference_v1.jsonl`) from approved QA rows.

Schema: `src/data/schema_pref.py` (`PreferencePair`).  
This does **not** rebuild QA. Each preference row is **derived** from one QA row via `base_example_id`.

QA construction: [`QA_GENERATION_PROTOCOL.md`](QA_GENERATION_PROTOCOL.md).

---

## Principle

1. Start from an approved QA row (`data_v1_pilot.jsonl` / `data_v1.jsonl`).
2. Build `prompt` by formatting evidence + question + instruction (from `configs/prompts/default.yaml`).
3. Set `chosen` = that row's `reference_answer`.
4. Write one deliberate bad answer as `rejected`, and label how it fails with `negative_type`.
5. Pilot default: **1 QA → 1 preference pair**. Same QA may get more pairs later (harder negatives).

Do not invent a new `answerability`. Slice negatives with existing QA labels:

- `answerability` = correct behavior
- `evidence_challenge` = why the item is hard
- `negative_type` = how `rejected` fails

---

## Row shape

| Field | Source |
|-------|--------|
| `id` | New id, e.g. `pref_0021` |
| `base_example_id` | QA `id`, e.g. `ex_0021` |
| `prompt` | Assembled from evidence + question + instruction |
| `chosen` | QA `reference_answer` |
| `rejected` | New bad answer (main new content) |
| `negative_type` | Failure mode of `rejected` |
| `dataset_version` | Same family as QA, e.g. `v1` |
| `metadata` | Optional (`creation_process`, `notes`) |

`prompt` is **concatenated**, not LLM-authored as a free rewrite of the task.

---

## Pilot negative map (required)

One pair per QA row for pilot. Decide `negative_type` from existing QA labels only:

`answerability` × `evidence_challenge` → one `negative_type`.

| # | `answerability` | `evidence_challenge` | Correct behavior (`chosen`) | Bad behavior (`rejected`) | `negative_type` |
|---|-----------------|----------------------|-----------------------------|---------------------------|-----------------|
| 1 | `answerable` | `[]` | Answer from evidence | Refuse / say don't know even though evidence answers | `over_refusal` |
| 2 | `answerable` | `["known_world_conflict"]` | Answer from evidence (even if anti-common-sense) | Prefer world knowledge over evidence | `memory_override` |
| 3 | `unanswerable` | `[]` | Refuse | Fabricate a plausible answer not in evidence | `hallucination` |
| 4 | `unanswerable` | `["distractor_entity"]` | Refuse | Answer using the distractor / wrong entity | `distractor_confusion` |
| 5 | `partial` | `["partial_evidence"]` | Answer supported part; refuse unsupported | Also fill the unsupported part | `over_complete` |

Decision order (same map in code):

1. If `answerable` and `known_world_conflict` → `memory_override`
2. Else if `answerable` → `over_refusal`
3. Else if `unanswerable` and `distractor_entity` → `distractor_confusion`
4. Else if `unanswerable` → `hallucination`
5. Else if `partial` → `over_complete`

Notes:

- `known_world_conflict` is **answerable**, not unanswerable.
- Do **not** use one generic hallucination template for all unanswerable rows: missing (`[]`) and `distractor_entity` need different `rejected` content.
- For pilot `partial`, use **`over_complete` only**. Optional later: same row + `over_refusal` (refuse even the supported part).

---

## Optional later negatives (not required for pilot)

| QA situation | Extra `rejected` idea | `negative_type` |
|--------------|----------------------|-----------------|
| `partial` | Refuse the whole question | `over_refusal` |
| `answerable` + `known_world_conflict` | Second pair with the other of `{memory_override, over_refusal}` | as labeled |
| any | Style-only bad answers, unrelated typos | avoid — not a grounding failure |

Keep hard negatives tied to the failure the challenge is testing.

---

## Style constraints

- `chosen` and `rejected` should be similarly fluent (avoid template `chosen` vs polished `rejected`, or DPO can reward style).
- Prefer QA rows that already finished Layer 2 paraphrase before building preference pairs.
- `rejected` must be wrong **for the stated reason** in `negative_type`, not merely different wording of `chosen`.

---

## Scale and pilot

| Split | Preference rows |
|-------|-----------------|
| Pilot `preference_v1_pilot.jsonl` | ~**50** (1:1 with `data_v1_pilot.jsonl`) |
| Full `preference_v1.jsonl` | ~**500** (1:1 with `data_v1.jsonl`), after full QA exists |

**IDs:** `pref_` + digits matching or derived from base example (e.g. `pref_0021` from `ex_0021`). Optional letter suffix for multiple pairs from one QA (`pref_0021a`).  
**Version:** `dataset_version: "v1"`.

**Pilot gate:** spot-check that each `negative_type` matches the QA slice above, and that `rejected` is actually wrong under evidence-only rules.

---

## Workflow

1. Freeze / approve the QA file used as base (pilot first).
2. For each QA row, assemble `prompt`, copy `chosen`, write `rejected` per the pilot map.
3. Validate with `PreferencePair` (`schema_pref.py`).
4. Human review a sample across all five pilot slices.
5. Scale with the same map when full QA exists; add optional second negatives only after the 1:1 map is solid.
