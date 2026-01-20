#!/usr/bin/env python3
# tokenize_wiki_chunks.py
# Safe Windows-compatible tokenization pipeline

import os, glob
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from multiprocessing import freeze_support

SRC_GLOB = "artifacts/processed/wiki_chunks/wiki_chunk_*.arrow"
OUT_DIR = "artifacts/processed/wiki_tokenized"
TOKENIZER = "artifacts/zia_tokenizer_60k"
MAX_LEN = 1024

def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(SRC_GLOB))
    print(f"[i] Found {len(files)} wiki chunks to tokenize")

    for f in tqdm(files):
        name = os.path.basename(f)
        out_f = os.path.join(OUT_DIR, name)
        if os.path.exists(out_f):
            print(f"[→] Skipping already tokenized {name}")
            continue

        ds = load_from_disk(f)

        def tokenize_fn(batch):
            texts = [f"User: {i}\nAssistant: {r}" for i, r in zip(batch["instruction"], batch["response"])]
            enc = tok(texts, truncation=True, padding=False, max_length=MAX_LEN)
            enc["labels"] = [ids[:] for ids in enc["input_ids"]]
            return enc

        # ⚠ Single-threaded map for Windows safety
        tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
        tokenized.save_to_disk(out_f)

    print("[✓] Tokenization complete.")

if __name__ == "__main__":
    freeze_support()
    main()
