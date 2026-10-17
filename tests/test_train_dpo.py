"""Tests for train_dpo.py's MaskRejectedEndCollator: it must leave out of the loss
exactly the end-of-turn token of each rejected response, and nothing else.
Token ids are small integers; no model or tokenizer is loaded.
"""

from __future__ import annotations

import torch
from trl.trainer.dpo_trainer import DataCollatorForPreference

from grounded_refusal.train.train_dpo import MaskRejectedEndCollator

PAD, END, NEWLINE = 0, 9, 8

# Two pairs with different prompt and response lengths, so the rows are padded.
# Each response ends with the end-of-turn token and a newline, as the chat template writes it.
EXAMPLES = [
    {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5, END, NEWLINE], "rejected_ids": [6, 7, 7, END, NEWLINE]},
    {"prompt_ids": [1, 2], "chosen_ids": [4, END, NEWLINE], "rejected_ids": [6, END, NEWLINE]},
]


def collate(collator_class, **kwargs):
    return collator_class(pad_token_id=PAD, **kwargs).torch_call(EXAMPLES)


def test_only_the_rejected_end_token_is_removed_from_the_loss():
    standard = collate(DataCollatorForPreference)
    masked = collate(MaskRejectedEndCollator, end_token_id=END)

    assert torch.equal(masked["input_ids"], standard["input_ids"])
    assert torch.equal(masked["attention_mask"], standard["attention_mask"])

    changed = (standard["completion_mask"] != masked["completion_mask"]).nonzero().tolist()
    # Rows 0-1 are the chosen sequences, rows 2-3 the rejected ones.
    assert changed == [[2, 3 + 3], [3, 2 + 1]]
    for row, column in changed:
        assert masked["input_ids"][row, column] == END
        assert standard["completion_mask"][row, column] == 1
        assert masked["completion_mask"][row, column] == 0


def test_the_newline_after_the_end_token_stays_in_the_loss():
    masked = collate(MaskRejectedEndCollator, end_token_id=END)
    assert masked["completion_mask"][2, 3 + 4] == 1
    assert masked["completion_mask"][3, 2 + 2] == 1
