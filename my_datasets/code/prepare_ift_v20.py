#!/usr/bin/env python3
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------- CONFIG ----------------
OUT_DIR = "datasets/processed/ift_v22"
TOK_PATH = "artifacts/zia_tokenizer_60k_clean"
MAX_LEN = 1024
TRAIN_SAMPLES = 12000
VAL_SAMPLES = 1000
# ---------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

print("[i] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(TOK_PATH, local_files_only=True)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "<pad>"})

print("[i] Loading OpenOrca...")
ds = load_dataset("Open-Orca/OpenOrca", split="train")

def format_sample(ex):
    instr = ex.get("question", "").strip()
    resp = ex.get("response", "").strip()
    if not instr or not resp:
        return None
    text = f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
    return text

def encode(text):
    ids = tok(
        text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    ).input_ids[0]
    labels = ids.clone()
    return ids, labels

train_ids, train_lbls = [], []
val_ids, val_lbls = [], []

print("[i] Processing samples...")
count = 0
for ex in tqdm(ds):
    text = format_sample(ex)
    if text is None:
        continue
    ids, lbls = encode(text)
    if count < TRAIN_SAMPLES:
        train_ids.append(ids)
        train_lbls.append(lbls)
    elif count < TRAIN_SAMPLES + VAL_SAMPLES:
        val_ids.append(ids)
        val_lbls.append(lbls)
    else:
        break
    count += 1

torch.save(
    {"input_ids": torch.stack(train_ids), "labels": torch.stack(train_lbls)},
    f"{OUT_DIR}/train.pt"
)
torch.save(
    {"input_ids": torch.stack(val_ids), "labels": torch.stack(val_lbls)},
    f"{OUT_DIR}/val.pt"
)

print(f"[✓] Saved {len(train_ids)} train and {len(val_ids)} val samples")
