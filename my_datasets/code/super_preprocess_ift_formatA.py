#!/usr/bin/env python3
"""
super_preprocess_ift_formatA.py
ZIA Unified IFT Preprocessor

This script:
- Loads raw datasets (json/jsonl/HF/ptxt)
- Normalizes into unified ChatML-style format:
      <|user|>\n{user_message}\n<|assistant|>\n{assistant_reply}
- Removes corrupted chars (Ċ, ĉ, stray unicode)
- Strips URLs, HTML, markup, wiki leftovers
- Deduplicates samples
- Filters short/long samples
- Tokenizes with ZIA tokenizer
- Pads/truncates to 4096 tokens
- Saves into .pt shards for ultra-fast training
"""

import os, json, glob, re, random
from dataclasses import dataclass
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# ================================
# CONFIG
# ================================
@dataclass
class CFG:
    RAW_DIR = "datasets/raw"  # Put raw JSON/JSONL/TXT/HF datasets here
    OUT_DIR = "datasets/processed/zia_superift_clean4k"

    TOKENIZER = "artifacts/zia_tokenizer_60k"
    MAX_LEN = 4096
    SHARD_SIZE = 5000

    # Filtering
    MIN_CHARS = 20
    MIN_TOKENS = 16
    MAX_INPUT_CHARS = 20000

CFG = CFG()

# ================================
# UTILITIES
# ================================
def clean_text(t: str):
    """Remove junk unicode, html, wiki/meta, bad bytes."""
    if not isinstance(t, str):
        return ""

    t = t.replace("\uFFFD", "")      # replacement char
    t = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", " ", t)  # remove control junk
    t = re.sub(r"Ċ|ĉ|č|Ġ|đ|ħ|ı|ň|ŧ", " ", t)         # model-bleed chars
    t = re.sub(r"<[^>]+>", " ", t)                   # HTML tags
    t = re.sub(r"\[\[[^\]]+\]\]", " ", t)            # wiki
    t = re.sub(r"https?://\S+", "", t)               # URLs
    t = re.sub(r"\s+", " ", t).strip()               # dedupe whitespace
    return t


def normalize_format(user_msg, assistant_msg):
    """Enforce unified ChatML-style format."""
    user_msg = clean_text(user_msg)
    assistant_msg = clean_text(assistant_msg)

    return (
        "<|user|>\n" +
        user_msg.strip() +
        "\n<|assistant|>\n" +
        assistant_msg.strip()
    )


def tokenize_and_pack(samples, tokenizer):
    """Tokenize, pad/truncate, and produce tensors."""
    input_ids = []
    labels = []

    for sample in tqdm(samples, desc="[Tokenizing]"):
        ids = tokenizer(sample, return_tensors="pt", truncation=True,
                        max_length=CFG.MAX_LEN).input_ids[0]

        # Prepare labels (shifted LM)
        lbl = ids.clone()
        lbl[lbl == tokenizer.pad_token_id] = -100

        # Pad manually to fixed length
        pad_len = CFG.MAX_LEN - ids.size(0)
        if pad_len > 0:
            pad_id = tokenizer.pad_token_id
            ids = torch.cat([ids, torch.full((pad_len,), pad_id)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100)])

        input_ids.append(ids.unsqueeze(0))
        labels.append(lbl.unsqueeze(0))

    return torch.cat(input_ids, dim=0), torch.cat(labels, dim=0)


def write_shards(ids, lbl, out_dir, split):
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)

    total = ids.size(0)
    shard_id = 0

    for i in range(0, total, CFG.SHARD_SIZE):
        shard_ids = ids[i:i+CFG.SHARD_SIZE]
        shard_lbl = lbl[i:i+CFG.SHARD_SIZE]

        shard_path = os.path.join(out_dir, split, f"shard_{shard_id:03d}")
        os.makedirs(shard_path, exist_ok=True)

        torch.save(shard_ids, os.path.join(shard_path, "input_ids.pt"))
        torch.save(shard_lbl, os.path.join(shard_path, "labels.pt"))
        shard_id += 1


# ================================
# MAIN — UNIFIED PIPELINE
# ================================
def load_raw_sources():
    """Load all raw datasets under RAW_DIR."""
    samples = []

    # JSON / JSONL
    for path in glob.glob(os.path.join(CFG.RAW_DIR, "**/*.json*"), recursive=True):
        try:
            data = load_dataset("json", data_files=path, split="train")
            for ex in data:
                u, a = ex.get("instruction") or ex.get("input"), ex.get("output")
                if u and a:
                    samples.append((u, a))
        except:
            pass

    # TXT or conversation dumps (if any)
    for path in glob.glob(os.path.join(CFG.RAW_DIR, "**/*.txt"), recursive=True):
        with open(path, "r", encoding="utf8") as f:
            lines = f.read().strip().split("\n\n")
        for chunk in lines:
            if ":" in chunk:
                parts = chunk.split("Assistant:")
                if len(parts) == 2:
                    u = parts[0].replace("User:", "").strip()
                    a = parts[1].strip()
                    samples.append((u, a))

    print(f"[i] Loaded raw sample pairs: {len(samples)}")
    return samples


def main():
    print("=== ZIA Unified Preprocessing (4K ChatML) ===")

    tokenizer = AutoTokenizer.from_pretrained(CFG.TOKENIZER, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    raw_pairs = load_raw_sources()
    cleaned = []

    # Normalize formats
    for u, a in tqdm(raw_pairs, desc="[Normalizing]"):
        if not u or not a:
            continue
        if len(u) < CFG.MIN_CHARS or len(a) < CFG.MIN_CHARS:
            continue
        if len(u) > CFG.MAX_INPUT_CHARS or len(a) > CFG.MAX_INPUT_CHARS:
            continue

        merged = normalize_format(u, a)
        cleaned.append(merged)

    print(f"[i] Cleaned normalized samples: {len(cleaned)}")

    # Deduplicate
    cleaned = list(set(cleaned))
    print(f"[i] After deduplication: {len(cleaned)}")

    # Shuffle + split
    random.shuffle(cleaned)
    val_size = max(1000, int(0.01 * len(cleaned)))
    val_samples = cleaned[:val_size]
    train_samples = cleaned[val_size:]

    print(f"[i] Train: {len(train_samples)}, Validation: {len(val_samples)}")

    # Tokenize + Pack
    train_ids, train_lbl = tokenize_and_pack(train_samples, tokenizer)
    val_ids, val_lbl = tokenize_and_pack(val_samples, tokenizer)

    # Write shards
    write_shards(train_ids, train_lbl, CFG.OUT_DIR, "train")
    write_shards(val_ids, val_lbl, CFG.OUT_DIR, "val")

    print("=== DONE: Clean 4K ChatML dataset READY ===")


if __name__ == "__main__":
    main()
