#!/usr/bin/env python3
"""
verify_shard_format.py
Quick checks for a single shard directory.
Usage:
    python my_datasets/code/verify_shard_format.py datasets/processed/hf_chatml_4096/shard_0000
"""

import sys, os, json
import torch
from transformers import PreTrainedTokenizerFast

def verify(shard_dir):
    tok = PreTrainedTokenizerFast.from_pretrained("artifacts/hf_tokenizer_mistral", local_files_only=True)
    pad_id = tok.pad_token_id or 0
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id

    in_path = os.path.join(shard_dir, "input_ids.pt")
    lbl_path = os.path.join(shard_dir, "labels.pt")
    meta_path = os.path.join(os.path.dirname(shard_dir), "meta.json")

    assert os.path.exists(in_path), "input_ids.pt missing"
    assert os.path.exists(lbl_path), "labels.pt missing"

    x = torch.load(in_path)
    y = torch.load(lbl_path)

    print("[i] loaded", in_path, "shape:", x.shape)
    print("[i] labels shape:", y.shape)

    # sample decode
    sample = x[0].tolist()
    print("sample first 40 ids:", sample[:40])
    print("decoded sample (first 512 tokens):")
    # decode safely (skip pad)
    try:
        dec = tok.decode([tid for tid in sample if tid != pad_id][:512], skip_special_tokens=False)
    except Exception as e:
        dec = f"<decode failed: {e}>"
    print(dec[:2000])

    # check tokens
    print("pad_id:", pad_id, "bos_id:", bos_id, "eos_id:", eos_id)
    # check shift mismatch between input and labels (we used same sequences, training script may expect shifting)
    # We'll show whether label vs input shifting matches expected property
    mismatches = 0
    limit = min(10, x.size(1)-1)
    for i in range(limit):
        if x[0, i+1].item() != y[0, i].item():
            mismatches += 1
    print("shift mismatches (first tokens):", mismatches, "(should be 0 if labels are shifted version)")

    if os.path.exists(meta_path):
        print("[i] meta:", json.load(open(meta_path)))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_shard_format.py <shard_dir>")
        sys.exit(1)
    verify(sys.argv[1])
