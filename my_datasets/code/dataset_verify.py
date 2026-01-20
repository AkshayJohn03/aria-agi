#!/usr/bin/env python3
"""
verify_and_fix_zia_dataset.py
Checks and fixes dataset format for ZIA IFT training.
Ensures 'input_ids', 'attention_mask', and 'labels' are present and consistent.
"""

import os
from datasets import load_from_disk, DatasetDict

DATASET_PATH = "artifacts/tokenized_dataset/zia_ift_v3"
OUTPUT_PATH = "artifacts/tokenized_dataset/zia_ift_v3_verified"

def verify_and_fix(ds_dict):
    fixed_dict = {}
    for split, ds in ds_dict.items():
        print(f"\n[🔍] Checking split: {split} ({len(ds)} samples)")
        n_ok, n_fix, n_drop = 0, 0, 0
        def _fix_record(ex):
            nonlocal n_ok, n_fix, n_drop
            if not all(k in ex for k in ["input_ids", "attention_mask", "labels"]):
                n_drop += 1
                return None
            L = len(ex["input_ids"])
            if len(ex["attention_mask"]) != L:
                ex["attention_mask"] = [1]*L
                n_fix += 1
            if len(ex["labels"]) != L:
                ex["labels"] = ex["input_ids"][:]
                n_fix += 1
            n_ok += 1
            return ex

        ds_fixed = ds.map(
            _fix_record,
            desc=f"Validating {split}",
            remove_columns=[c for c in ds.column_names if c not in ["input_ids", "attention_mask", "labels"]],
        ).filter(lambda x: x is not None)

        print(f"   OK: {n_ok} | Fixed: {n_fix} | Dropped: {n_drop}")
        fixed_dict[split] = ds_fixed
    return DatasetDict(fixed_dict)

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    ds_dict = load_from_disk(DATASET_PATH)
    verified = verify_and_fix(ds_dict)
    verified.save_to_disk(OUTPUT_PATH)
    print(f"\n✅ Verified dataset saved to: {OUTPUT_PATH}")
