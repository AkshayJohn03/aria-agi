#!/usr/bin/env python3
# prepare_alpaca.py
# Download Alpaca dataset and clean for IFT

import os
from datasets import load_dataset, DatasetDict

OUT_DIR = "my_datasets/processed/alpaca_ift"

def main():
    print("[i] Downloading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca")

    # Convert into "text" field format for your collator
    def format_example(ex):
        inst = ex["instruction"].strip()
        resp = ex["output"].strip()
        # Chat-style text
        return {
            "text": f"User: {inst}\nAssistant: {resp}</s>"
        }

    print("[i] Cleaning and formatting...")
    ds = ds.map(format_example, remove_columns=ds["train"].column_names)

    # Split into train/val (90/10)
    ds = ds["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({
        "train": ds["train"],
        "validation": ds["test"]
    })

    os.makedirs(OUT_DIR, exist_ok=True)
    dataset.save_to_disk(OUT_DIR)
    print(f"[✓] Saved Alpaca IFT dataset → {OUT_DIR}")
    print(dataset)

if __name__ == "__main__":
    main()
