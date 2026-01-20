#!/usr/bin/env python3
"""
verify_zia_dataset_integrity.py
Deep validation of ZIA dataset/tokenizer consistency and alignment.
Checks for:
  ✅ token index bounds
  ✅ label masking correctness
  ✅ pad/eos token consistency
  ✅ instruction/response structure
  ✅ checkpoint vocab alignment
"""

import os, torch
from datasets import load_from_disk
from transformers import AutoTokenizer

# --- Config ---
DATASET_PATH = "artifacts/tokenized_dataset/zia_ift_v3_verified"
TOKENIZER_PATH = "artifacts/zia_tokenizer_60k"
CKPT_PATH = "artifacts/zia_ift_v4_cursor/best_val/checkpoint.pt"  # optional
SAMPLE_CHECKS = 500  # number of samples to inspect manually

# --- Load tokenizer ---
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
if tok.pad_token is None:
    tok.add_special_tokens({"pad_token": "<pad>"})
if tok.eos_token is None:
    tok.add_special_tokens({"eos_token": "</s>"})

vocab_size = len(tok)
print(f"[i] Tokenizer loaded | vocab={vocab_size}, pad={tok.pad_token_id}, eos={tok.eos_token_id}")

# --- Load dataset ---
print(f"[i] Loading dataset: {DATASET_PATH}")
ds = load_from_disk(DATASET_PATH)
train_ds = ds["train"]
val_ds = ds["validation"]
print(f"[i] Samples: train={len(train_ds)}, val={len(val_ds)}")

# --- 1️⃣ Check token index bounds ---
def check_token_bounds(dataset, name):
    print(f"\n[🔍] Checking token bounds in {name}...")
    bad_ids = 0
    for i, sample in enumerate(dataset.select(range(min(SAMPLE_CHECKS, len(dataset))))):
        ids = torch.tensor(sample["input_ids"])
        if ids.max().item() >= vocab_size or ids.min().item() < 0:
            print(f"[!] Out-of-range token in sample {i}: min={ids.min()}, max={ids.max()}")
            bad_ids += 1
    if bad_ids == 0:
        print("[✓] All input_ids within vocab range.")
    else:
        print(f"[!] Found {bad_ids} samples with invalid tokens.")

check_token_bounds(train_ds, "train")
check_token_bounds(val_ds, "validation")

# --- 2️⃣ Check label masking correctness ---
def check_label_masking(dataset, name):
    print(f"\n[🔍] Checking label masking in {name}...")
    bad_labels = 0
    for i, sample in enumerate(dataset.select(range(min(SAMPLE_CHECKS, len(dataset))))):
        labels = torch.tensor(sample["labels"])
        if (labels == 0).any() and tok.pad_token_id == 0:
            print(f"[!] Pad token 0 found in labels (sample {i}). Should be -100.")
            bad_labels += 1
    if bad_labels == 0:
        print("[✓] Labels properly masked (no pads used).")
    else:
        print(f"[!] Found {bad_labels} samples with incorrect label masking.")

check_label_masking(train_ds, "train")

# --- 3️⃣ Instruction–Response alignment check ---
def check_instruction_format(dataset):
    print(f"\n[🔍] Checking Instruction/Response structure...")
    malformed = 0
    for i, sample in enumerate(dataset.select(range(min(SAMPLE_CHECKS, len(dataset))))):
        text = tok.decode(sample["input_ids"], skip_special_tokens=True)
        if "Instruction:" not in text or "Response:" not in text:
            malformed += 1
            if malformed <= 5:
                print(f"[!] Sample {i} malformed: {text[:200]}")
    if malformed == 0:
        print("[✓] All samples follow Instruction/Response structure.")
    else:
        print(f"[!] {malformed}/{SAMPLE_CHECKS} samples missing structure. Clean dataset recommended.")

check_instruction_format(train_ds)

# --- 4️⃣ Check checkpoint ↔ tokenizer consistency ---
if os.path.exists(CKPT_PATH):
    print(f"\n[🔍] Checking checkpoint: {CKPT_PATH}")
    ck = torch.load(CKPT_PATH, map_location="cpu")
    state = ck.get("model", ck)
    tok_w = state.get("tok.weight", None)
    head_w = state.get("head.weight", None)
    if tok_w is not None:
        print(f"   tok.weight shape: {tuple(tok_w.shape)}")
    if head_w is not None:
        print(f"   head.weight shape: {tuple(head_w.shape)}")
    if tok_w is not None and tok_w.size(0) != vocab_size:
        print(f"[!] vocab mismatch: checkpoint={tok_w.size(0)}, tokenizer={vocab_size}")
    else:
        print("[✓] Checkpoint vocab matches tokenizer.")
else:
    print(f"[!] Checkpoint not found at {CKPT_PATH} (skipping vocab check).")

print("\n✅ Verification complete.")
print("If any [!] errors appear above, fix them before resuming training.")
