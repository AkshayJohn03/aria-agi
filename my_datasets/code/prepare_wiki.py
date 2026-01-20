#!/usr/bin/env python3
# prepare_wiki_v21.py — clean English Wikipedia (script-free)

import os, re
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------- CONFIG ----------------
OUT_DIR = "datasets/processed/wiki_clean_1024"
TOKENIZER_PATH = "artifacts/zia_tokenizer_60k_clean"

SEQ_LEN = 1024
SHARD_SIZE = 2000        # sequences per shard
MAX_DOCS = 150_000       # enough for stabilization
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("[i] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "<pad>"})

print("[i] Loading Wikimedia Wikipedia (EN)...")
ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True
)

def clean_text(text: str) -> str:
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

buffer_x, buffer_y = [], []
shard_idx = 0
doc_count = 0

def save_shard(x, y, idx):
    path = Path(OUT_DIR) / f"train_shard_{idx:04d}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(x), path / "input_ids.pt")
    torch.save(torch.stack(y), path / "labels.pt")
    print(f"[✓] Saved shard {idx} ({len(x)} samples)")

print("[i] Processing documents...")

for row in tqdm(ds):
    if doc_count >= MAX_DOCS:
        break

    text = clean_text(row.get("text", ""))
    if len(text) < 800:
        continue

    ids = tok(
        text,
        add_special_tokens=False,
        truncation=False
    ).input_ids

    for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN):
        chunk = ids[i:i + SEQ_LEN]
        buffer_x.append(torch.tensor(chunk))
        buffer_y.append(torch.tensor(chunk))

        if len(buffer_x) >= SHARD_SIZE:
            save_shard(buffer_x, buffer_y, shard_idx)
            buffer_x, buffer_y = [], []
            shard_idx += 1

    doc_count += 1

if buffer_x:
    save_shard(buffer_x, buffer_y, shard_idx)

print("[✓] Wikipedia preparation complete.")
