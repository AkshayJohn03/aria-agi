#!/usr/bin/env python3
"""
verify_dataset_samples.py

Prints 10 random reconstructed samples from the processed fixed-4k dataset.
"""

import os, random, torch
from transformers import AutoTokenizer

DATASET_DIR = "datasets/processed/zia_superift_clean4k/train"
TOKENIZER = "artifacts/zia_tokenizer_60k"

NUM_SAMPLES = 10

def load_random_shard(path):
    shards = [os.path.join(path, d) for d in os.listdir(path)]
    shards = [s for s in shards if os.path.isdir(s)]

    if not shards:
        raise RuntimeError("No shards found!")

    return random.choice(shards)

def decode_text(ids, tok):
    ids = ids.tolist()
    if tok.pad_token_id in ids:
        ids = ids[:ids.index(tok.pad_token_id)]
    return tok.decode(ids, skip_special_tokens=False)

def main():
    print("=== VERIFY DATASET SAMPLES ===")

    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)

    shard = load_random_shard(DATASET_DIR)
    print(f"[i] Random shard selected: {shard}")

    ids_path = os.path.join(shard, "input_ids.pt")
    ids = torch.load(ids_path)

    n = ids.size(0)
    print(f"[i] Shard contains {n} samples")

    print("\n----- RANDOM SAMPLES -----\n")

    for _ in range(NUM_SAMPLES):
        idx = random.randint(0, n - 1)
        decoded = decode_text(ids[idx], tok)

        print(f"[Sample #{idx}]")
        print(decoded)
        print("\n-----------------------------\n")

    print("=== DONE ===")


if __name__ == "__main__":
    main()
