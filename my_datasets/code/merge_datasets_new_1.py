#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge merged_parts + new Wikipedia chunk into Arrow format
"""

import os
import glob
import time
from tqdm import tqdm
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Features, Value, Sequence

# Paths
merged_dir = "my_datasets/organized/merged"
wiki_chunk = "my_datasets/organized/wikipedia_chunk_4000000.jsonl"
out_dir = "my_datasets/processed/arrow_dataset_v2"

# Define a unified feature schema to handle inconsistent columns
unified_features = Features({
    "messages": Sequence(
        feature=Features({
            "role": Value("string"),
            "content": Value("string")
        })
    ),
    "source": Value("string"),
    "type": Value("string"),
})


def load_jsonl(path, features):
    return load_dataset("json", data_files=path, split="train", features=features)


def main():
    # Load all merged parts
    merged_parts = glob.glob(os.path.join(merged_dir, "*.jsonl"))
    print(f"[i] Found {len(merged_parts)} merged parts")
    print(f"[i] Using unified schema: {unified_features}")

    datasets = []
    start_time = time.time()
    for p in tqdm(sorted(merged_parts), desc="Loading merged datasets"):
        datasets.append(load_jsonl(p, features=unified_features))
    
    # Add new Wikipedia chunk
    print(f" - loading {wiki_chunk}")
    datasets.append(load_jsonl(wiki_chunk, features=unified_features))
    print(f"[✓] All datasets loaded. Time taken: {time.time() - start_time:.2f} seconds")

    # Concatenate
    print("\n[i] Starting dataset concatenation...")
    start_time = time.time()
    full = concatenate_datasets(datasets)
    print(f"[✓] Dataset concatenation complete. Time taken: {time.time() - start_time:.2f} seconds")
    print(f"[✓] Total samples after merge: {len(full):,}")

    # Train/val split (95/5)
    print("\n[i] Creating train/validation split...")
    start_time = time.time()
    split = full.train_test_split(test_size=0.05, seed=42)
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })
    print(f"[✓] Train/validation split complete. Time taken: {time.time() - start_time:.2f} seconds")

    # Save to Arrow
    print(f"\n[i] Saving Arrow dataset to {out_dir}...")
    start_time = time.time()
    dataset.save_to_disk(out_dir)
    print(f"[✓] Saved Arrow dataset to {out_dir}. Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()