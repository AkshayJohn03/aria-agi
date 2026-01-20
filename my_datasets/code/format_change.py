#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
format_change_resume.py
- Efficiently shards unified JSONL → Arrow with resumability.
"""

import os
from datasets import load_dataset

# --- Config ---
jsonl_path = "normalized_merged_temp.jsonl"   # unified 15.7M dataset
out_root = "my_datasets/processed/arrow_normalized_v1"
train_dir = os.path.join(out_root, "train")
test_dir = os.path.join(out_root, "test")

NUM_TRAIN_SHARDS = 122     # total train shards
TEST_RATIO = 0.05          # 5% validation split
SEED = 42                  # reproducibility
NUM_PROC = os.cpu_count()  # parallelism


def save_shard(dataset, out_dir, idx, num_shards):
    """Save one shard safely via slicing."""
    shard_dir = os.path.join(out_dir, f"shard_{idx:03d}")
    if os.path.exists(shard_dir):
        print(f"[✓] Skipping train shard {idx}/{num_shards} (already exists)")
        return

    total = len(dataset)
    start = (total * idx) // num_shards
    end = (total * (idx + 1)) // num_shards
    shard = dataset.select(range(start, end))

    print(f"[i] Saving train shard {idx}/{num_shards} → rows {start:,}–{end:,} ({len(shard):,} rows)")
    shard.save_to_disk(shard_dir)
    print(f"[+] Saved train shard {idx}/{num_shards}")


def main():
    print("[i] Loading unified JSONL...")
    ds = load_dataset(
        "json",
        data_files=jsonl_path,
        split="train",
        num_proc=NUM_PROC,  # parallel reading
    )

    print("[i] Shuffling dataset...")
    ds = ds.shuffle(seed=SEED)   # ⚠️ no num_proc here

    print("[i] Splitting (95% train, 5% validation)...")
    split_ds = ds.train_test_split(test_size=TEST_RATIO, seed=SEED)
    train_ds, test_ds = split_ds["train"], split_ds["test"]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # --- TRAIN SHARDS ---
    existing = [d for d in os.listdir(train_dir) if d.startswith("shard_")]
    existing_count = len(existing)
    print(f"[i] Detected {existing_count} existing train shards in {train_dir}")

    for i in range(existing_count, NUM_TRAIN_SHARDS):
        save_shard(train_ds, train_dir, i, NUM_TRAIN_SHARDS)

    # --- TEST SPLIT ---
    if os.listdir(test_dir):
        print("[✓] Test split already exists, skipping")
    else:
        print("[i] Saving TEST split...")
        test_ds.save_to_disk(os.path.join(test_dir, "test"))
        print("[+] Saved test split")

    print("[✓] Done! Dataset ready at", out_root)


if __name__ == "__main__":
    main()
